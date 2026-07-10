from __future__ import annotations

from pathlib import Path
import time
from typing import Any, cast

from visualact_r.agents.core import MockVisualAgent
from visualact_r.agents.qwen_adapter import (
    QwenVLMAdapter,
    QwenVLMConfig,
    build_qlora_model,
    run_qlora_sft,
)
from visualact_r import __version__
from visualact_r.artifacts import (
    load_candidate_cache,
    load_crops,
    load_inference_records,
    load_records,
    load_split_image_ids,
    load_traces,
    save_candidate_cache,
    save_crops,
    save_traces,
)
from visualact_r.cef.core import evaluate_development_gate, fit_cef, make_crops
from visualact_r.config import load_config
from visualact_r.data.build import build_task_records
from visualact_r.agents.sft import materialize_crop, prepare_sft_stage
from visualact_r.data.splits import build_splits, validate_split_integrity
from visualact_r.detector.synthetic import SyntheticColorDetector
from visualact_r.detector.ultralytics_adapter import (
    RawCandidateExtractionUnsupported,
    UltralyticsRawCandidateAdapter,
)
from visualact_r.evaluation.metrics import (
    evaluate_candidate,
    paired_bootstrap_delta,
    review_capacity_scenarios,
    slice_metrics,
)
from visualact_r.evaluation.release import release_decision
from visualact_r.rctr.core import RCTRModel, apply_review_routes, fit_rctr, select_review_routes
from visualact_r.schemas import (
    Action,
    AgentStep,
    AgentTrace,
    CropRecord,
    InferenceRecord,
    TaskRecord,
)
from visualact_r.utils import (
    ensure_dir,
    read_json,
    sha256_directory,
    sha256_file,
    stable_hash,
    utc_now,
    write_json,
    write_jsonl,
    read_jsonl,
    write_table,
)


def _split(records: list[TaskRecord], split: str) -> list[TaskRecord]:
    return [x for x in records if x.split == split]


def build_data_stage(
    annotation_path: str, images_dir: str, output: str, config_path: str | None = None
) -> dict[str, Any]:
    cfg = load_config(config_path)
    root = ensure_dir(output)
    records = build_task_records(annotation_path, images_dir, cfg, root)
    records = build_splits(records, cfg, str(root))
    integrity = validate_split_integrity(records, cfg)
    inference_path = root / "inference_records.jsonl"
    report = {
        "status": integrity["status"],
        "record_count": len(records),
        "split_integrity": integrity,
        "task_records_path": str(root / "task_records.jsonl"),
        "inference_records_path": str(inference_path),
        "inference_records_sha256": sha256_file(inference_path),
        "config_hash": stable_hash(cfg),
    }
    write_json(root / "build_data_report.json", report)
    return report


def _validate_pre_nms_cache(
    cache: dict[str, tuple[list[Any], list[Any]]],
    expected_image_ids: set[str],
    allowed_empty_image_ids: set[str] | None = None,
) -> dict[str, Any]:
    allowed_empty = allowed_empty_image_ids or set()
    missing = sorted((expected_image_ids - allowed_empty) - set(cache))
    unexpected_empty = sorted(allowed_empty - expected_image_ids)
    invalid_images: list[str] = []
    raw_count = 0
    nms_count = 0
    covered_image_count = 0
    for image_id, (raw, nms) in cache.items():
        raw_count += len(raw)
        nms_count += len(nms)
        if image_id in allowed_empty and not raw and not nms:
            continue
        if raw or nms:
            covered_image_count += 1
        valid = bool(raw) and all(candidate.pre_nms for candidate in raw)
        valid = valid and len(raw) >= len(nms) and all(not candidate.pre_nms for candidate in nms)
        valid = valid and all(candidate.image_id == image_id for candidate in [*raw, *nms])
        if not valid:
            invalid_images.append(image_id)
    detector_miss_count = len(allowed_empty.intersection(expected_image_ids))
    valid_status = not missing and not invalid_images and not unexpected_empty
    return {
        "status": "PASS" if valid_status else "HOLD",
        "missing_image_ids": missing,
        "invalid_image_ids": invalid_images,
        "unexpected_detector_miss_image_ids": unexpected_empty,
        "record_count": len(expected_image_ids),
        "candidate_covered_count": covered_image_count,
        "detector_miss_count": detector_miss_count,
        "candidate_coverage_rate": covered_image_count / len(expected_image_ids)
        if expected_image_ids
        else 0.0,
        "raw_candidate_count": raw_count,
        "nms_candidate_count": nms_count,
        "raw_candidate_extraction_validated": valid_status,
        "contract": (
            "raw entries must be pre_nms=true; selected NMS entries must be pre_nms=false; "
            "image coverage may exclude explicitly recorded detector misses"
        ),
    }


def _load_detector_miss_ids(path: str | None) -> set[str]:
    if path is None:
        return set()
    return {str(row["image_id"]) for row in read_jsonl(path)}


def cache_detector_stage(
    records_path: str,
    output: str,
    backend: str,
    weights: str | None = None,
    config_path: str | None = None,
    confidence: float = 0.10,
    imgsz: int = 640,
    device: str | None = None,
    allow_post_nms_fallback: bool = False,
    detector_miss_policy: str = "fail",
    miss_retry_imgsz: int | None = None,
    miss_retry_confidence: float | None = None,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    records = load_records(records_path)
    root = ensure_dir(output)
    if detector_miss_policy not in {"fail", "record"}:
        raise ValueError("detector_miss_policy must be one of: fail, record")
    if miss_retry_imgsz is not None and miss_retry_imgsz <= 0:
        raise ValueError("miss_retry_imgsz must be positive when provided")
    if miss_retry_confidence is not None and not 0.0 <= miss_retry_confidence <= 1.0:
        raise ValueError("miss_retry_confidence must be in [0, 1] when provided")
    if backend == "synthetic":
        detector: Any = SyntheticColorDetector()
    elif backend == "ultralytics":
        if not weights:
            raise ValueError("--weights is required for the ultralytics backend")
        detector = UltralyticsRawCandidateAdapter(
            weights=weights,
            imgsz=imgsz,
            confidence=confidence,
            device=device,
            allow_post_nms_fallback=allow_post_nms_fallback,
        )
    else:
        raise ValueError(f"Unsupported detector backend: {backend}")
    retry_detector: Any | None = None
    if backend == "ultralytics" and miss_retry_imgsz is not None:
        retry_detector = UltralyticsRawCandidateAdapter(
            weights=cast(str, weights),
            imgsz=miss_retry_imgsz,
            confidence=confidence if miss_retry_confidence is None else miss_retry_confidence,
            device=device,
            allow_post_nms_fallback=allow_post_nms_fallback,
        )
    cache: dict[str, tuple[list[Any], list[Any]]] = {}
    failures: list[dict[str, Any]] = []
    detector_misses: list[dict[str, Any]] = []
    detector_retry_image_ids: list[str] = []
    post_nms_fallback_image_ids: list[str] = []
    for record in records:
        try:
            cache[record.image_id] = detector.predict(record)
            if getattr(detector, "last_used_post_nms_fallback", False):
                post_nms_fallback_image_ids.append(record.image_id)
        except RawCandidateExtractionUnsupported as exc:
            miss = {
                "image_id": record.image_id,
                "image_path": record.image_path,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            if retry_detector is not None:
                try:
                    cache[record.image_id] = retry_detector.predict(record)
                    detector_retry_image_ids.append(record.image_id)
                    if getattr(retry_detector, "last_used_post_nms_fallback", False):
                        post_nms_fallback_image_ids.append(record.image_id)
                    continue
                except RawCandidateExtractionUnsupported as retry_exc:
                    miss = {
                        **miss,
                        "retry_error_type": type(retry_exc).__name__,
                        "retry_error": str(retry_exc),
                        "retry_imgsz": miss_retry_imgsz,
                        "retry_confidence": confidence
                        if miss_retry_confidence is None
                        else miss_retry_confidence,
                    }
            if detector_miss_policy == "record":
                detector_misses.append(miss)
                cache[record.image_id] = ([], [])
                continue
            failures.append(miss)
            break
        except Exception as exc:  # noqa: BLE001 - report non-miss detector failures.
            failures.append(
                {
                    "image_id": record.image_id,
                    "image_path": record.image_path,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            break
    report_base = {
        "backend": backend,
        "weights": weights,
        "confidence": confidence,
        "imgsz": imgsz,
        "device": device,
        "allow_post_nms_fallback": allow_post_nms_fallback,
        "detector_miss_policy": detector_miss_policy,
        "miss_retry_imgsz": miss_retry_imgsz,
        "miss_retry_confidence": miss_retry_confidence,
        "detector_retry_count": len(detector_retry_image_ids),
        "detector_retry_image_ids": detector_retry_image_ids[:20],
        "post_nms_fallback_count": len(post_nms_fallback_image_ids),
        "post_nms_fallback_image_ids": post_nms_fallback_image_ids[:20],
        "detector_miss_count": len(detector_misses),
        "detector_miss_image_ids": [x["image_id"] for x in detector_misses[:20]],
        "detector_misses_path": str(root / "detector_misses.jsonl"),
        "detector_failures": failures,
        "config_hash": stable_hash(cfg),
    }
    if failures:
        report = {
            **report_base,
            "status": "FAIL",
            "record_count": len(records),
            "processed_record_count": len(cache),
            "raw_candidate_extraction_validated": False,
            "contract": "detector must return mappable raw candidates for every requested image",
        }
        write_json(root / "detector_cache_report.json", report)
        return report
    write_jsonl(root / "detector_misses.jsonl", detector_misses)
    save_candidate_cache(
        cache, root / "detector_raw_candidates.jsonl", root / "detector_nms_candidates.jsonl"
    )
    validation = _validate_pre_nms_cache(
        cache,
        {record.image_id for record in records},
        {x["image_id"] for x in detector_misses},
    )
    report = {**validation, **report_base}
    if report["status"] != "PASS":
        write_json(root / "detector_cache_report.json", report)
        return report
    write_json(root / "detector_cache_report.json", report)
    return report


def train_cef_stage(
    records_path: str,
    raw_path: str,
    nms_path: str,
    output: str,
    config_path: str | None = None,
    detector_misses_path: str | None = None,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    records = load_records(records_path)
    cache = load_candidate_cache(raw_path, nms_path)
    detector_miss_ids = _load_detector_miss_ids(detector_misses_path)
    cache_validation = _validate_pre_nms_cache(
        cache, {record.image_id for record in records}, detector_miss_ids
    )
    if cache_validation["status"] != "PASS":
        raise RuntimeError(
            "CEF training requires a validated pre-NMS cache; "
            f"missing={len(cache_validation['missing_image_ids'])}, "
            f"invalid={len(cache_validation['invalid_image_ids'])}, "
            f"unexpected_misses={len(cache_validation['unexpected_detector_miss_image_ids'])}"
        )
    usable_records = [record for record in records if record.image_id not in detector_miss_ids]
    train = _split(usable_records, "train")
    if not train:
        raise ValueError("No train records available")
    model = fit_cef(train, cache, cfg)
    dev = _split(usable_records, "dev")
    development_gate = evaluate_development_gate(model, dev, cache, cfg)
    model.development_gate = development_gate
    root = ensure_dir(output)
    model.save(root / "cef_model.pkl")
    variants = {"C3": "nms", "C4": "raw_ranker", "C5": "cef"}
    rows = []
    for candidate_id, variant in variants.items():
        crops_by_image: dict[str, list[CropRecord]] = {}
        for record in usable_records:
            raw, nms = cache[record.image_id]
            crops_by_image[record.image_id] = make_crops(
                record, raw, nms, variant, cfg, model if variant in {"raw_ranker", "cef"} else None
            )
        save_crops(root / f"crop_records_{candidate_id}.jsonl", crops_by_image, candidate_id)
        rows.extend(
            [
                {**crop.to_dict(), "candidate_id": candidate_id}
                for crops in crops_by_image.values()
                for crop in crops
            ]
        )
    write_jsonl(root / "crop_records_all.jsonl", rows)
    write_json(root / "cef_development_gate.json", development_gate)
    report = {
        "status": "PASS" if development_gate.get("status") == "PASS" else "HOLD",
        "stage_execution_status": "PASS",
        "train_record_count": len(train),
        "dev_record_count": len(dev),
        "usable_record_count": len(usable_records),
        "detector_miss_excluded_count": len(detector_miss_ids),
        "detector_misses_path": detector_misses_path,
        "directional_corrector_models": len(model.corrector.models),
        "ranker_mode": "learned" if model.ranker.model is not None else "constant_fallback",
        "raw_ranker_mode": "learned" if model.raw_ranker.model is not None else "constant_fallback",
        "development_gate": development_gate,
        "pre_nms_cache_validation": cache_validation,
        "config_hash": stable_hash(cfg),
    }
    write_json(root / "cef_training_report.json", report)
    return report


def rollout_mock_stage(
    records_path: str,
    crops_path: str,
    candidate_id: str,
    output: str,
    config_path: str | None = None,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    records = load_records(records_path)
    crops = load_crops(crops_path, candidate_id)
    agent = MockVisualAgent(candidate_id)
    traces = [agent.rollout(record, crops.get(record.image_id, []), cfg) for record in records]
    root = ensure_dir(output)
    save_traces(root / f"agent_traces_{candidate_id}.jsonl", traces)
    report = {
        "status": "PASS" if all(trace.schema_valid for trace in traces) else "HOLD",
        "backend": "fixture_mock_only",
        "candidate_id": candidate_id,
        "trace_count": len(traces),
        "schema_valid_rate": sum(x.schema_valid for x in traces) / len(traces) if traces else 0.0,
    }
    write_json(root / f"rollout_{candidate_id}_report.json", report)
    return report


def prepare_sft_dataset_stage(
    records_path: str,
    crops_path: str | None,
    candidate_id: str,
    split: str,
    output: str,
    max_crops: int = 3,
) -> dict[str, Any]:
    return prepare_sft_stage(
        records_path=records_path,
        crops_path=crops_path,
        candidate_id=candidate_id,
        split=split,
        output=output,
        max_crops=max_crops,
    )


def qwen_training_stage(
    train_sft_path: str,
    dev_sft_path: str | None,
    output: str,
    config_path: str | None = None,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    q = cfg.get("qwen", {})
    qcfg = QwenVLMConfig(
        model_id=q.get("model_id", "Qwen/Qwen3-VL-4B-Instruct"),
        revision=q.get("revision", "main"),
        load_in_4bit=bool(q.get("load_in_4bit", True)),
        quant_type=q.get("quant_type", "nf4"),
        lora_rank=int(q.get("lora_rank", 16)),
        lora_alpha=int(q.get("lora_alpha", 32)),
        lora_dropout=float(q.get("lora_dropout", 0.05)),
        target_modules=q.get("target_modules", "all-linear"),
        max_new_tokens=int(q.get("max_new_tokens", 192)),
        max_images=int(q.get("max_images", 4)),
    )
    raw_steps = q.get("max_steps")
    return run_qlora_sft(
        train_examples_path=train_sft_path,
        dev_examples_path=dev_sft_path,
        output_dir=output,
        config=qcfg,
        train_batch_size=int(q.get("train_batch_size", 1)),
        gradient_accumulation_steps=int(q.get("gradient_accumulation_steps", 8)),
        learning_rate=float(q.get("learning_rate", 1e-4)),
        num_train_epochs=float(q.get("num_train_epochs", 1.0)),
        max_steps=int(raw_steps) if raw_steps is not None else None,
        gradient_checkpointing=bool(q.get("gradient_checkpointing", True)),
    )


def qwen_training_preflight_stage(
    output: str, config_path: str | None = None, execute_model_load: bool = False
) -> dict[str, Any]:
    """Produces a reproducible QLoRA plan; optional model loading verifies local GPU compatibility.

    It deliberately does not label a plan as training success. A real optimizer run requires independently
    generated train-only image/crop SFT data and a completed run manifest.
    """
    cfg = load_config(config_path)
    q = cfg.get("qwen", {})
    qcfg = QwenVLMConfig(
        model_id=q.get("model_id", "Qwen/Qwen3-VL-4B-Instruct"),
        revision=q.get("revision", "main"),
        load_in_4bit=bool(q.get("load_in_4bit", True)),
        quant_type=q.get("quant_type", "nf4"),
        lora_rank=int(q.get("lora_rank", 16)),
        lora_alpha=int(q.get("lora_alpha", 32)),
        lora_dropout=float(q.get("lora_dropout", 0.05)),
        target_modules=q.get("target_modules", "all-linear"),
        max_new_tokens=int(q.get("max_new_tokens", 192)),
        max_images=int(q.get("max_images", 4)),
    )
    root = ensure_dir(output)
    status = "PLAN_READY"
    error = None
    if execute_model_load:
        try:
            model, processor = build_qlora_model(qcfg)
            # Do not persist a full base model. Save only adapter/config work after explicit real training.
            del model, processor
            status = "MODEL_LOAD_VALIDATED_NOT_TRAINED"
        except (
            Exception
        ) as exc:  # hardware/model dependency preflight should be observable, not hidden
            status = "MODEL_LOAD_HOLD"
            error = repr(exc)
    plan = {
        "status": status,
        "candidate_id": "C2/C3/C5_QWEN_QLORA",
        "base_model": qcfg.model_id,
        "revision": qcfg.revision,
        "quantization": {"load_in_4bit": qcfg.load_in_4bit, "quant_type": qcfg.quant_type},
        "lora": {
            "rank": qcfg.lora_rank,
            "alpha": qcfg.lora_alpha,
            "dropout": qcfg.lora_dropout,
            "target_modules": qcfg.target_modules,
        },
        "execute_model_load": execute_model_load,
        "error": error,
        "claim_boundary": "Plan or model-load validation is not a completed QLoRA training result.",
        "config_hash": stable_hash(cfg),
    }
    write_json(root / "qwen_qlora_training_plan.json", plan)
    return plan


def train_rctr_stage(
    records_path: str,
    traces_path: str,
    crops_path: str,
    output: str,
    candidate_id: str = "C5",
    config_path: str | None = None,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    records = load_records(records_path)
    train = _split(records, "train")
    train_ids = {record.image_id for record in train}
    all_candidate_traces = load_traces(traces_path, candidate_id)
    traces = [trace for trace in all_candidate_traces if trace.image_id in train_ids]
    by_image: dict[str, AgentTrace] = {}
    duplicates: list[str] = []
    for trace in traces:
        if trace.image_id in by_image:
            duplicates.append(trace.image_id)
        by_image[trace.image_id] = trace
    missing = sorted(train_ids - set(by_image))
    if duplicates or missing:
        raise ValueError(
            f"RCTR train trace contract failed: duplicates={len(set(duplicates))}, missing={len(missing)}"
        )
    crops = load_crops(crops_path, candidate_id)
    model = fit_rctr(
        train,
        list(by_image.values()),
        crops,
        n_reference=int(cfg["rctr"]["max_reference_library"]),
        neighbor_count=int(cfg["rctr"]["n_reference"]),
    )
    root = ensure_dir(output)
    model_path = root / "rctr_model.pkl"
    model.save(model_path)
    report = {
        "status": "PASS",
        "reference_count": len(model.reference_trace_ids),
        "reference_prefix_count": len(model.reference_prefix_ids),
        "feature_names": model.feature_names,
        "train_trace_count": len(traces),
        "train_record_count": len(train),
        "train_trace_coverage_rate": len(traces) / len(train) if train else 0.0,
        "references_train_only": True,
        "prefix_only_contract": True,
        "source_records_sha256": sha256_file(records_path),
        "source_traces_sha256": sha256_file(traces_path),
        "source_crops_sha256": sha256_file(crops_path),
        "rctr_model_sha256": sha256_file(model_path),
        "config_hash": stable_hash(cfg),
    }
    write_json(root / "rctr_training_report.json", report)
    return report


def evaluate_stage(
    records_path: str,
    traces_path: str,
    crops_path: str,
    candidate_id: str,
    output: str,
    config_path: str | None = None,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    records = [x for x in load_records(records_path) if x.split == "final"]
    traces = [
        x
        for x in load_traces(traces_path, candidate_id)
        if x.image_id in {r.image_id for r in records}
    ]
    crops = load_crops(crops_path, candidate_id)
    metrics, rows = evaluate_candidate(records, traces, crops, candidate_id, cfg)
    root = ensure_dir(output)
    write_json(root / f"metrics_{candidate_id}.json", metrics)
    write_table(
        rows, root / f"predictions_{candidate_id}.csv", root / f"predictions_{candidate_id}.parquet"
    )
    write_table(
        slice_metrics(rows),
        root / f"slice_metrics_{candidate_id}.csv",
        root / f"slice_metrics_{candidate_id}.parquet",
    )
    risk = {x.image_id: 1.0 - x.final_confidence for x in traces}
    write_table(
        review_capacity_scenarios(rows, risk),
        root / f"decision_scenarios_{candidate_id}.csv",
        root / f"decision_scenarios_{candidate_id}.parquet",
    )
    return metrics


def write_project_creation_handoff(
    root: str | Path,
    config: dict[str, Any],
    preflight_status: str,
    source_preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(root)
    handoff = {
        "schema_version": "1.1",
        "project_id": "visualact-r",
        "project_version": __version__,
        "project_depth": {
            "foundation_complete": True,
            "industry_baseline_complete": True,
            "advanced_technique_complete": True,
            "specialty_module_complete": True,
            "product_layer_complete": True,
        },
        "candidate_portfolio": [
            {
                "candidate_id": "C0",
                "type": "sanity",
                "algorithm": "detector-zone heuristic",
                "status": "fixture_executed",
            },
            {
                "candidate_id": "C1",
                "type": "industry",
                "algorithm": "calibrated logistic detector triage",
                "status": "fixture_executed",
            },
            {
                "candidate_id": "C2/C3/C5",
                "type": "advanced",
                "algorithm": "Qwen3-VL 4-bit QLoRA training and bounded rollout interface",
                "status": "implemented_not_gpu_executed",
            },
            {
                "candidate_id": "C4",
                "type": "specialty_ablation",
                "algorithm": "learned raw-candidate crop selector",
                "status": "fixture_executed",
            },
            {
                "candidate_id": "C5",
                "type": "specialty",
                "algorithm": "ABC-derived CEF",
                "status": "fixture_executed",
            },
            {
                "candidate_id": "C6",
                "type": "production",
                "algorithm": "CEF + matched-capacity RCTR review gate",
                "status": "fixture_executed_with_controlled_routing_contract",
            },
        ],
        "specialty_transfer": [
            {
                "source_method": "ABC/Probability Bbox",
                "adapted_name": "Critical Evidence Field",
                "adaptation_type": "domain_extension",
                "implementation_path": "src/visualact_r/cef/core.py",
                "limitations": ["fixture metrics are not BDD/VLM claims"],
            },
            {
                "source_method": "PHT reference-state alignment",
                "adapted_name": "RCTR",
                "adaptation_type": "derived",
                "implementation_path": "src/visualact_r/rctr/core.py",
                "limitations": ["requires real trace validation before claims"],
            },
        ],
        "champion_selection": {
            "predictive_champion": None,
            "production_champion": None,
            "selection_criteria": [
                "sealed C3/C5 frozen-final comparison",
                "paired bootstrap minimum sample requirement",
                "matched-capacity C5 confidence guard versus C6 RCTR",
            ],
            "tradeoffs": [],
        },
        "outputs": {
            "numeric": [
                "metrics.json",
                "candidate_comparison.csv",
                "slice_metrics.csv",
                "decision_scenarios.csv",
            ],
            "visualizations": ["figures/*.png", "figures/*.svg"],
            "reports": ["report.md", "report.html"],
            "cards": ["docs/MODEL_CARD_TEMPLATE.md", "docs/DATA_CARD_TEMPLATE.md"],
            "api_or_cli_examples": [
                "visualact-r fixture-run",
                "visualact-r preflight",
                "visualact-r rollout-qwen",
                "visualact-r route-review",
                "visualact-r compare-final",
            ],
        },
        "experiment_lineage": {
            "config_hash": stable_hash(config),
            "source_preflight_status": preflight_status,
            "source_preflight": source_preflight or {},
        },
        "claim_boundary": "Fixture artifacts validate the repository contract only. QLoRA and BDD-derived claims require separate completed local runs.",
        "created_at_utc": utc_now(),
    }
    write_json(root / "project_creation_handoff.json", handoff)
    return handoff


def _final_records(records_path: str) -> list[TaskRecord]:
    records = [record for record in load_records(records_path) if record.split == "final"]
    if not records:
        raise ValueError("No frozen-final records are available")
    return records


def _final_candidate_traces(
    traces_path: str,
    candidate_id: str,
    final_image_ids: set[str],
) -> list[Any]:
    traces = [
        trace
        for trace in load_traces(traces_path, candidate_id)
        if trace.image_id in final_image_ids
    ]
    by_image: dict[str, Any] = {}
    duplicates: list[str] = []
    for trace in traces:
        if trace.image_id in by_image:
            duplicates.append(trace.image_id)
        by_image[trace.image_id] = trace
    if duplicates:
        raise ValueError(
            f"Duplicate {candidate_id} trace outputs for final images: {sorted(set(duplicates))}"
        )
    missing = sorted(final_image_ids - set(by_image))
    if missing:
        raise ValueError(
            f"Incomplete {candidate_id} frozen-final trace coverage: missing {len(missing)} image IDs"
        )
    return list(by_image.values())


def route_review_stage(
    records_path: str,
    c5_traces_path: str,
    c5_crops_path: str,
    rctr_model_path: str,
    output: str,
    config_path: str | None = None,
) -> dict[str, Any]:
    """Create matched-capacity confidence and RCTR review routes without reading final labels.

    The frozen-final records are used only to select the immutable image set. No action label, supporting-box,
    or outcome field is read in this stage. This separation makes the routing artifact inspectable before evaluation.
    """
    cfg = load_config(config_path)
    final_ids = load_split_image_ids(records_path, "final")
    if not final_ids:
        raise ValueError("No frozen-final records are available")
    c5_traces = _final_candidate_traces(c5_traces_path, "C5", final_ids)
    if not c5_traces:
        raise ValueError("No C5 traces matched frozen-final records")
    crops = load_crops(c5_crops_path, "C5")
    training_report_path = Path(rctr_model_path).parent / "rctr_training_report.json"
    if not training_report_path.exists():
        raise FileNotFoundError("RCTR training report is required beside the model artifact")
    training_report = read_json(training_report_path)
    if (
        training_report.get("status") != "PASS"
        or training_report.get("references_train_only") is not True
        or training_report.get("prefix_only_contract") is not True
        or training_report.get("rctr_model_sha256") != sha256_file(rctr_model_path)
    ):
        raise RuntimeError("RCTR training lineage contract is not valid")
    model = RCTRModel.load(rctr_model_path)
    review_capacity = float(cfg["rctr"]["review_capacity"])
    rctr_risk: dict[str, float] = {}
    rctr_refs: dict[str, list[str]] = {}
    for trace in c5_traces:
        risk, refs = model.risk(
            trace,
            crops.get(trace.image_id, []),
            n_reference=int(cfg["rctr"]["n_reference"]),
            prefix_len=max(0, len(trace.steps) - 1),
        )
        rctr_risk[trace.image_id] = risk
        rctr_refs[trace.image_id] = refs
    confidence_risk = {trace.image_id: 1.0 - float(trace.final_confidence) for trace in c5_traces}
    confidence_routes = select_review_routes(c5_traces, confidence_risk, review_capacity)
    rctr_routes = select_review_routes(c5_traces, rctr_risk, review_capacity)
    c5_guard = apply_review_routes(
        c5_traces,
        confidence_routes,
        confidence_risk,
        {},
        candidate_id="C5_CONFIDENCE_GUARD",
        reason_code="confidence_only_capacity_route",
    )
    c6 = apply_review_routes(
        c5_traces,
        rctr_routes,
        rctr_risk,
        rctr_refs,
        candidate_id="C6",
        reason_code="rctr_prefix_risk_capacity_route",
    )
    root = ensure_dir(output)
    save_traces(root / "agent_traces_C5_CONFIDENCE_GUARD.jsonl", c5_guard)
    save_traces(root / "agent_traces_C6.jsonl", c6)
    write_json(
        root / "rctr_reference_library.json",
        {
            "reference_prefix_count": len(model.reference_prefix_ids),
            "reference_trace_count": len(model.reference_trace_ids),
            "feature_names": model.feature_names,
            "references_train_only": True,
            "rctr_model_sha256": sha256_file(rctr_model_path),
        },
    )
    report = {
        "status": "PASS",
        "scope": "frozen_final_predecision_only",
        "frozen_final_record_count": len(final_ids),
        "c5_trace_count": len(c5_traces),
        "c5_trace_coverage_rate": len(c5_traces) / len(final_ids),
        "review_capacity": review_capacity,
        "confidence_guard_route_count": len(confidence_routes),
        "rctr_route_count": len(rctr_routes),
        "confidence_guard_route_ids": sorted(confidence_routes),
        "rctr_route_ids": sorted(rctr_routes),
        "rctr_model_sha256": sha256_file(rctr_model_path),
        "rctr_training_report_sha256": sha256_file(training_report_path),
        "label_fields_read": 0,
        "terminal_fields_used_by_rctr": 0,
        "config_hash": stable_hash(cfg),
        "claim_boundary": "This routing artifact is pre-decision only; it is not an outcome evaluation.",
    }
    write_json(root / "rctr_routing_report.json", report)
    return report


def _counterfactual_not_executed_rows(final: list[TaskRecord]) -> list[dict[str, Any]]:
    return [
        {
            "image_id": record.image_id,
            "candidate_id": "C5",
            "status": "NOT_EXECUTED",
            "reason": "Real counterfactual crop removal requires repeated VLM inference and is not inferred from offline traces.",
            "scope": "frozen_final_artifact_placeholder_no_causal_claim",
        }
        for record in final
    ]


def _validate_real_qwen_rollout_protocol(
    trace_paths: dict[str, str],
    crop_paths: dict[str, str],
    records_path: str,
) -> dict[str, Any]:
    """Validate sealed C3/C4/C5 Qwen lineage before final comparison.

    The shared adapter must be trained only on C2 full-scene examples. Final rollout
    reads the dedicated inference artifact, while labels remain in task_records.jsonl
    solely for later offline evaluation. Crop/config fingerprints must match their
    actual files so a stale report cannot satisfy the protocol.
    """
    reports: dict[str, dict[str, Any]] = {}
    inference_path = Path(records_path).parent / "inference_records.jsonl"
    if not inference_path.exists():
        raise FileNotFoundError(
            "Real final comparison requires the dedicated inference_records.jsonl adjacent to task_records.jsonl"
        )
    expected_inference_sha = sha256_file(inference_path)
    adapter_fingerprints: set[str] = set()
    config_hashes: set[str] = set()
    model_signatures: set[tuple[Any, ...]] = set()
    for candidate_id, trace_path in trace_paths.items():
        report_path = Path(trace_path).parent / f"rollout_{candidate_id}_report.json"
        if not report_path.exists():
            raise FileNotFoundError(
                f"Missing Qwen rollout report for {candidate_id}: {report_path}"
            )
        report = read_json(report_path)
        expected_crops_sha = sha256_file(crop_paths[candidate_id])
        required = (
            report.get("status") == "PASS"
            and report.get("candidate_id") == candidate_id
            and report.get("split") == "final"
            and report.get("label_fields_read") == 0
            and report.get("ground_truth_fields_read") == 0
            and report.get("source_inference_records_sha256") == expected_inference_sha
            and report.get("source_crops_sha256") == expected_crops_sha
            and report.get("adapter_fingerprint")
            and report.get("max_image_pixels") is not None
        )
        if not required:
            raise RuntimeError(f"Invalid sealed Qwen rollout lineage for {candidate_id}")
        adapter_fingerprints.add(str(report["adapter_fingerprint"]))
        config_hashes.add(str(report.get("config_hash")))
        quantization = report.get("quantization", {})
        model_signatures.add(
            (
                report.get("base_model"),
                report.get("revision"),
                quantization.get("load_in_4bit"),
                quantization.get("quant_type"),
                report.get("max_image_pixels"),
                report.get("adapter_dir"),
            )
        )
        reports[candidate_id] = report
    if len(adapter_fingerprints) != 1:
        raise RuntimeError(
            "C3/C4/C5 frozen comparison requires the same complete adapter fingerprint"
        )
    if len(config_hashes) != 1 or len(model_signatures) != 1:
        raise RuntimeError("C3/C4/C5 frozen comparison requires identical Qwen model/configuration")
    # All reports must point to a real shared full-scene-only training manifest.
    adapter_dir = Path(next(iter(reports.values()))["adapter_dir"])
    training_manifest_path = adapter_dir.parent / "qwen_qlora_training_manifest.json"
    if not training_manifest_path.exists():
        raise FileNotFoundError(
            f"Missing shared adapter training manifest: {training_manifest_path}"
        )
    training_manifest = read_json(training_manifest_path)
    if (
        training_manifest.get("status") != "TRAINED_LOCAL_GPU"
        or training_manifest.get("training_protocol") != "shared_full_scene_only"
        or training_manifest.get("adapter_fingerprint") != next(iter(adapter_fingerprints))
    ):
        raise RuntimeError(
            "Shared adapter was not trained under the required C2 full-scene-only protocol"
        )
    return {
        "shared_adapter_fingerprint": next(iter(adapter_fingerprints)),
        "shared_adapter_training_manifest_sha256": sha256_file(training_manifest_path),
        "source_inference_records_sha256": expected_inference_sha,
        "rollout_report_sha256": {
            candidate: sha256_file(Path(path).parent / f"rollout_{candidate}_report.json")
            for candidate, path in trace_paths.items()
        },
    }


def compare_final_stage(
    records_path: str,
    c3_traces_path: str,
    c3_crops_path: str,
    c5_traces_path: str,
    c5_crops_path: str,
    c5_guard_traces_path: str,
    c6_traces_path: str,
    cef_development_gate_path: str,
    rctr_routing_report_path: str,
    output: str,
    scope: str = "bdd100k_derived_internal_frozen_final",
    config_path: str | None = None,
    c4_traces_path: str | None = None,
    c4_crops_path: str | None = None,
) -> dict[str, Any]:
    """Evaluate the sealed C3/C5/C5-confidence/C6 comparison on frozen-final records.

    This command intentionally consumes only already-created traces and crops. It does not train, tune thresholds,
    select references, or regenerate labels. The resulting output is the canonical evidence package for any later
    release/claim gate.
    """
    if scope not in {"bdd100k_derived_internal_frozen_final", "fixture_only"}:
        raise ValueError("scope must be bdd100k_derived_internal_frozen_final or fixture_only")
    cfg = load_config(config_path)
    final = _final_records(records_path)
    final_ids = {record.image_id for record in final}
    traces = {
        "C3": _final_candidate_traces(c3_traces_path, "C3", final_ids),
        "C5": _final_candidate_traces(c5_traces_path, "C5", final_ids),
        "C5_CONFIDENCE_GUARD": _final_candidate_traces(
            c5_guard_traces_path, "C5_CONFIDENCE_GUARD", final_ids
        ),
        "C6": _final_candidate_traces(c6_traces_path, "C6", final_ids),
    }
    c3_crops = load_crops(c3_crops_path, "C3")
    c5_crops = load_crops(c5_crops_path, "C5")
    c4_crops: dict[str, list[CropRecord]] | None = None
    if scope == "bdd100k_derived_internal_frozen_final" and (
        not c4_traces_path or not c4_crops_path
    ):
        raise ValueError(
            "Real frozen-final comparison requires C4 raw-selector traces and crops for attribution"
        )
    if c4_traces_path and c4_crops_path:
        traces["C4"] = _final_candidate_traces(c4_traces_path, "C4", final_ids)
        c4_crops = load_crops(c4_crops_path, "C4")
    rollout_protocol: dict[str, Any] | None = None
    if scope == "bdd100k_derived_internal_frozen_final":
        assert c4_traces_path is not None
        assert c4_crops_path is not None
        rollout_protocol = _validate_real_qwen_rollout_protocol(
            {"C3": c3_traces_path, "C4": c4_traces_path, "C5": c5_traces_path},
            {"C3": c3_crops_path, "C4": c4_crops_path, "C5": c5_crops_path},
            records_path,
        )
    if not Path(cef_development_gate_path).exists():
        raise FileNotFoundError(f"Missing CEF development gate: {cef_development_gate_path}")
    if not Path(rctr_routing_report_path).exists():
        raise FileNotFoundError(f"Missing RCTR routing report: {rctr_routing_report_path}")
    cef_gate = read_json(cef_development_gate_path)
    routing_report = read_json(rctr_routing_report_path)
    if (
        routing_report.get("status") != "PASS"
        or routing_report.get("label_fields_read") != 0
        or routing_report.get("terminal_fields_used_by_rctr") != 0
    ):
        raise RuntimeError("RCTR routing pre-decision contract is not valid")
    if cef_gate.get("status") != "PASS" and scope != "fixture_only":
        raise RuntimeError(
            "CEF development gate did not pass; refusing real frozen-final comparison"
        )
    crops_by_candidate = {
        "C3": c3_crops,
        "C5": c5_crops,
        "C5_CONFIDENCE_GUARD": c5_crops,
        "C6": c5_crops,
    }
    if c4_crops is not None:
        crops_by_candidate["C4"] = c4_crops
    metrics_by_candidate: dict[str, dict[str, Any]] = {}
    candidate_rows: dict[str, list[dict[str, Any]]] = {}
    all_slice_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    for candidate_id, candidate_traces in traces.items():
        metric, rows = evaluate_candidate(
            final, candidate_traces, crops_by_candidate[candidate_id], candidate_id, cfg
        )
        metrics_by_candidate[candidate_id] = metric
        candidate_rows[candidate_id] = rows
        all_slice_rows.extend(
            [{**row, "candidate_id": candidate_id} for row in slice_metrics(rows)]
        )
        if candidate_id == "C6":
            risk = {trace.image_id: float(trace.risk_score or 0.0) for trace in candidate_traces}
        else:
            risk = {
                trace.image_id: 1.0 - float(trace.pre_route_confidence or trace.final_confidence)
                for trace in candidate_traces
            }
        scenario_rows.extend(
            [{**row, "candidate_id": candidate_id} for row in review_capacity_scenarios(rows, risk)]
        )
    comparisons = {
        "C5_minus_C3_hard_grounded_action_success": paired_bootstrap_delta(
            candidate_rows["C3"],
            candidate_rows["C5"],
            "grounded_action_success",
            final_group="hard",
            iterations=int(cfg["evaluation"]["bootstrap_iterations"]),
            seed=int(cfg["project"]["seed"]),
            min_samples=int(cfg["evaluation"]["min_paired_final_samples"]),
        ),
        "C5_minus_C3_all_critical_evidence_recall": paired_bootstrap_delta(
            candidate_rows["C3"],
            candidate_rows["C5"],
            "critical_evidence_recall",
            iterations=int(cfg["evaluation"]["bootstrap_iterations"]),
            seed=int(cfg["project"]["seed"]),
            min_samples=int(cfg["evaluation"]["min_paired_final_samples"]),
        ),
        "C6_minus_C5_confidence_guard_grounded_action_success": paired_bootstrap_delta(
            candidate_rows["C5_CONFIDENCE_GUARD"],
            candidate_rows["C6"],
            "grounded_action_success",
            iterations=int(cfg["evaluation"]["bootstrap_iterations"]),
            seed=int(cfg["project"]["seed"]),
            min_samples=int(cfg["evaluation"]["min_paired_final_samples"]),
        ),
    }
    if "C4" in candidate_rows:
        comparisons["C5_minus_C4_all_grounded_action_success"] = paired_bootstrap_delta(
            candidate_rows["C4"],
            candidate_rows["C5"],
            "grounded_action_success",
            iterations=int(cfg["evaluation"]["bootstrap_iterations"]),
            seed=int(cfg["project"]["seed"]),
            min_samples=int(cfg["evaluation"]["min_paired_final_samples"]),
        )
    root = ensure_dir(output)
    write_json(root / "metrics.json", metrics_by_candidate)
    write_table(
        list(metrics_by_candidate.values()),
        root / "candidate_comparison.csv",
        root / "candidate_comparison.parquet",
    )
    paired_rows = [{"comparison_id": name, **value} for name, value in comparisons.items()]
    write_json(root / "paired_comparisons.json", comparisons)
    write_table(paired_rows, root / "paired_comparisons.csv", root / "paired_comparisons.parquet")
    write_table(all_slice_rows, root / "slice_metrics.csv", root / "slice_metrics.parquet")
    write_table(scenario_rows, root / "decision_scenarios.csv", root / "decision_scenarios.parquet")
    write_table(
        _counterfactual_not_executed_rows(final),
        root / "counterfactual_evidence.csv",
        root / "counterfactual_evidence.parquet",
    )
    write_json(root / "cef_development_gate.json", cef_gate)
    write_json(root / "rctr_routing_report.json", routing_report)
    routing_root = Path(rctr_routing_report_path).parent
    reference_path = routing_root / "rctr_reference_library.json"
    if reference_path.exists():
        write_json(root / "rctr_reference_library.json", read_json(reference_path))
    else:
        write_json(
            root / "rctr_reference_library.json",
            {"status": "HOLD", "reason": "RCTR reference artifact missing from routing stage"},
        )
    from visualact_r.pipeline import _rctr_controlled_stress_test

    controlled = _rctr_controlled_stress_test()
    write_json(root / "rctr_controlled_stress_test.json", controlled)
    split_manifest_path = Path(records_path).parent / "split_manifest.json"
    split_manifest = read_json(split_manifest_path) if split_manifest_path.exists() else None
    operational = {
        "scope": scope,
        "candidate_trace_counts": {key: len(value) for key, value in traces.items()},
        "trace_coverage_rates": {key: len(value) / len(final) for key, value in traces.items()},
        "model_training_metrics": "not inferred; read separate local training manifests",
        "rctr_predecision_routing_status": routing_report.get("status"),
        "rctr_controlled_stress_status": controlled.get("status"),
        "counterfactual_status": "NOT_EXECUTED",
        "counterfactual_claim_boundary": "No causal grounding conclusion without repeated VLM intervention runs.",
    }
    write_json(root / "operational_metrics.json", operational)
    manifest = {
        "run_id": stable_hash(
            {
                "scope": scope,
                "records_path": str(Path(records_path).resolve()),
                "final_ids": sorted(final_ids),
                "config_hash": stable_hash(cfg),
                "created": utc_now(),
            }
        )[:16],
        "scope": scope,
        "created_at_utc": utc_now(),
        "config_hash": stable_hash(cfg),
        "dataset_fingerprint": stable_hash(sorted(final_ids)),
        "frozen_final_image_count": len(final),
        "source_records_path": str(Path(records_path).resolve()),
        "source_records_sha256": sha256_file(records_path),
        "split_manifest": split_manifest,
        "split_manifest_sha256": sha256_file(split_manifest_path)
        if split_manifest_path.exists()
        else None,
        "gt_fields_in_inference_contract": 0,
        "candidate_ids": list(traces),
        "rctr_prefix_feature_contract": "pre-final trace steps only; terminal action/confidence excluded",
        "comparison_contract": "C3/C4/C5 use the same sealed Qwen adapter; C5 confidence guard/C6 are matched-capacity routes on C5 traces.",
        "qwen_rollout_protocol": rollout_protocol,
        "input_trace_sha256": {
            "C3": sha256_file(c3_traces_path),
            "C5": sha256_file(c5_traces_path),
            "C5_CONFIDENCE_GUARD": sha256_file(c5_guard_traces_path),
            "C6": sha256_file(c6_traces_path),
            **({"C4": sha256_file(c4_traces_path)} if c4_traces_path else {}),
        },
        "input_crop_sha256": {
            "C3": sha256_file(c3_crops_path),
            "C5": sha256_file(c5_crops_path),
            **({"C4": sha256_file(c4_crops_path)} if c4_crops_path else {}),
        },
        "cef_development_gate_sha256": sha256_file(cef_development_gate_path),
        "rctr_routing_report_sha256": sha256_file(rctr_routing_report_path),
    }
    write_json(root / "run_manifest.json", manifest)
    release = release_decision(metrics_by_candidate, scope, cfg, comparisons)
    from visualact_r.reporting import generate_reports

    generate_reports(root, release, manifest)
    return {"status": "PASS", "release": release, "manifest": manifest}


def _qwen_config_from_project_config(
    config: dict[str, Any], adapter_dir: str | None = None
) -> QwenVLMConfig:
    q = config.get("qwen", {})
    return QwenVLMConfig(
        model_id=q.get("model_id", "Qwen/Qwen3-VL-4B-Instruct"),
        revision=q.get("revision", "main"),
        load_in_4bit=bool(q.get("load_in_4bit", True)),
        quant_type=q.get("quant_type", "nf4"),
        lora_rank=int(q.get("lora_rank", 16)),
        lora_alpha=int(q.get("lora_alpha", 32)),
        lora_dropout=float(q.get("lora_dropout", 0.05)),
        target_modules=q.get("target_modules", "all-linear"),
        max_new_tokens=int(q.get("max_new_tokens", 192)),
        max_images=int(q.get("max_images", 4)),
        max_image_pixels=(
            int(q["max_image_pixels"]) if q.get("max_image_pixels") is not None else None
        ),
        adapter_dir=adapter_dir,
    )


def _qwen_crop_catalog(record: InferenceRecord, crops: list[CropRecord]) -> str:
    rows: list[str] = []
    for crop in sorted(crops, key=lambda item: item.selection_rank):
        box = crop.box_xyxy
        rows.append(
            f"{crop.crop_id}: category={crop.category}; normalized_box="
            f"({box.x1 / record.width:.3f},{box.y1 / record.height:.3f},"
            f"{box.x2 / record.width:.3f},{box.y2 / record.height:.3f}); "
            f"detector_score={crop.coverage_proxy:.3f}"
        )
    return "\n".join(rows) if rows else "No candidate evidence crops are available."


def _qwen_trace_for_record(
    adapter: QwenVLMAdapter,
    record: InferenceRecord,
    crops: list[CropRecord],
    materialized_dir: Path,
) -> AgentTrace:
    """Run a bounded CEF-retrieval-plus-Qwen finalization protocol without labels or GT boxes.

    CEF deterministically retrieves the ranked evidence crops. The VLM receives the full scene plus that evidence
    set and returns the final typed decision. This keeps the C3/C5 comparison attributable to the evidence policy,
    rather than letting an untrained free-form tool-selection policy decide whether to look at any crop at all.
    """
    start = time.perf_counter()
    selected = sorted(crops, key=lambda item: item.selection_rank)
    allowed = {crop.crop_id: crop for crop in selected}
    steps: list[AgentStep] = []
    image_paths = [record.image_path]
    for crop in selected:
        crop_path = materialize_crop(record, crop, materialized_dir, "QWEN_ROLLOUT")
        image_paths.append(crop_path)
        steps.append(
            AgentStep(
                step=len(steps) + 1,
                typed_action="INSPECT_REGION",
                region_id=crop.crop_id,
                decision=None,
                confidence=float(crop.coverage_proxy),
                schema_valid=True,
                reason_code="cef_policy_selected_evidence",
                latency_ms=0.0,
            )
        )
    catalog = _qwen_crop_catalog(record, selected)
    try:
        final_prompt = (
            f"{record.prompt}\nThe first image is the full scene. The following images are detector/CEF "
            f"evidence crops in this order:\n{catalog}\n"
            "Return compact JSON only with tool=FINALIZE, a valid decision, confidence in [0,1], "
            "evidence_ids drawn only from the supplied crop IDs, and reason_code."
        )
        response = adapter.generate_typed_action(image_paths, final_prompt)
        if response["tool"] == "ROUTE_REVIEW":
            final_decision: Action = "REVIEW"
        elif response["tool"] == "FINALIZE":
            final_decision = cast(Action, str(response["decision"]))
        else:
            raise ValueError("Qwen response did not produce FINALIZE or ROUTE_REVIEW")
        evidence_ids = [str(value) for value in response.get("evidence_ids", [])]
        if any(value not in allowed for value in evidence_ids):
            raise ValueError(
                "model finalized with evidence IDs outside the request-scoped candidate set"
            )
        steps.append(
            AgentStep(
                step=len(steps) + 1,
                typed_action="FINALIZE" if response["tool"] == "FINALIZE" else "ROUTE_REVIEW",
                region_id=None,
                decision=final_decision,
                confidence=float(response.get("confidence", 0.0)),
                schema_valid=True,
                reason_code=str(response.get("reason_code", "qwen_finalize")),
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        )
        return AgentTrace(
            trace_id=f"{record.image_id}:QWEN",
            image_id=record.image_id,
            task_id=record.task_id,
            candidate_id="QWEN",
            visible_crop_ids=[crop.crop_id for crop in selected],
            steps=steps,
            final_decision=final_decision,
            final_confidence=float(response.get("confidence", 0.0)),
            schema_valid=True,
            latency_ms=(time.perf_counter() - start) * 1000,
            final_evidence_ids=evidence_ids,
        )
    except Exception as exc:
        # Fail closed. Preserve an auditable terminal review for downstream evaluation.
        steps.append(
            AgentStep(
                step=max(1, len(steps) + 1),
                typed_action="ROUTE_REVIEW",
                region_id=None,
                decision="REVIEW",
                confidence=0.0,
                schema_valid=False,
                reason_code=f"qwen_fail_closed:{type(exc).__name__}",
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        )
        return AgentTrace(
            trace_id=f"{record.image_id}:QWEN",
            image_id=record.image_id,
            task_id=record.task_id,
            candidate_id="QWEN",
            visible_crop_ids=[crop.crop_id for crop in selected],
            steps=steps,
            final_decision="REVIEW",
            final_confidence=0.0,
            schema_valid=False,
            latency_ms=(time.perf_counter() - start) * 1000,
            final_evidence_ids=[],
        )


def rollout_qwen_stage(
    records_path: str,
    crops_path: str | None,
    candidate_id: str,
    split: str,
    output: str,
    adapter_dir: str | None = None,
    config_path: str | None = None,
) -> dict[str, Any]:
    """Run a local, bounded Qwen visual-agent rollout on a sealed split.

    It uses image files and detector/CEF crop metadata only. GT objects and offline action labels remain unread.
    The command refuses any split outside train/dev/calibration/final, and final use is suitable only after model
    selection and routing thresholds have been locked on earlier splits.
    """
    if split not in {"train", "dev", "calibration", "final"}:
        raise ValueError("split must be train, dev, calibration, or final")
    cfg = load_config(config_path)
    records = load_inference_records(records_path, split=split)
    if not records:
        raise ValueError(f"No records found for split={split!r}")
    crops = load_crops(crops_path, candidate_id) if crops_path else {}
    adapter = QwenVLMAdapter(_qwen_config_from_project_config(cfg, adapter_dir=adapter_dir))
    root = ensure_dir(output)
    materialized = ensure_dir(root / "inference_crops")
    traces: list[AgentTrace] = []
    for record in records:
        trace = _qwen_trace_for_record(
            adapter, record, crops.get(record.image_id, []), materialized
        )
        traces.append(
            AgentTrace(
                trace_id=f"{record.image_id}:{candidate_id}",
                image_id=trace.image_id,
                task_id=trace.task_id,
                candidate_id=candidate_id,
                visible_crop_ids=trace.visible_crop_ids,
                steps=trace.steps,
                final_decision=trace.final_decision,
                final_confidence=trace.final_confidence,
                schema_valid=trace.schema_valid,
                latency_ms=trace.latency_ms,
                final_evidence_ids=trace.final_evidence_ids,
            )
        )
    save_traces(root / f"agent_traces_{candidate_id}.jsonl", traces)
    report = {
        "status": "PASS" if all(trace.schema_valid for trace in traces) else "HOLD",
        "scope": f"local_qwen_rollout_{split}",
        "candidate_id": candidate_id,
        "split": split,
        "trace_count": len(traces),
        "schema_valid_rate": sum(trace.schema_valid for trace in traces) / len(traces),
        "label_fields_read": 0,
        "ground_truth_fields_read": 0,
        "base_model": adapter.config.model_id,
        "revision": adapter.config.revision,
        "adapter_dir": adapter_dir,
        "quantization": {
            "load_in_4bit": adapter.config.load_in_4bit,
            "quant_type": adapter.config.quant_type,
        },
        "config_hash": stable_hash(cfg),
        "source_inference_records_sha256": sha256_file(records_path),
        "source_crops_sha256": sha256_file(crops_path) if crops_path else None,
        "adapter_fingerprint": sha256_directory(adapter_dir) if adapter_dir else None,
        "max_image_pixels": adapter.config.max_image_pixels,
        "inference_record_contract": (
            "Dedicated inference_records.jsonl excludes action labels, supporting IDs, GT boxes, "
            "hard-case metadata, and pHash split metadata. The loader rejects forbidden fields."
        ),
        "claim_boundary": "A local rollout is not a frozen performance result. Final claims require compare-final and release gates.",
    }
    write_json(root / f"rollout_{candidate_id}_report.json", report)
    return report
