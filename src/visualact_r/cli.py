from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from visualact_r.config import config_template_names, load_config, write_config_template
from visualact_r.data.preflight import preflight_external_bdd
from visualact_r.data.extract import extract_bdd_assets
from visualact_r.data.bdd_direct import (
    BDDDirectLayoutError,
    build_direct_bdd_records,
    inspect_extracted_bdd_layout,
)
from visualact_r.agents.feasibility import LocalQLoRAFeasibilityConfig, run_local_qlora_feasibility
from visualact_r.pipeline import run_fixture_pipeline
from visualact_r.provenance import assert_installed_distribution
from visualact_r.stages import (
    build_data_stage,
    cache_detector_stage,
    compare_final_stage,
    evaluate_stage,
    qwen_training_preflight_stage,
    qwen_training_stage,
    prepare_sft_dataset_stage,
    rollout_mock_stage,
    rollout_qwen_stage,
    route_review_stage,
    train_cef_stage,
    train_rctr_stage,
)
from visualact_r.utils import read_json, write_json


def cmd_fixture(args: argparse.Namespace) -> int:
    result = run_fixture_pipeline(args.output, args.config, args.seed)
    print(result["release"]["status"])
    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    report = preflight_external_bdd(args.data_root, args.labels_zip, args.shard, args.output)
    print(report["status"])
    return 0 if report["status"] == "PASS" else 2


def cmd_extract_bdd(args: argparse.Namespace) -> int:
    report = extract_bdd_assets(
        args.data_root, args.labels_zip, args.shard, args.output, overwrite=bool(args.overwrite)
    )
    print(report["status"])
    return 0 if report.get("status") == "PASS" else 2


def cmd_bdd_direct_audit(args: argparse.Namespace) -> int:
    try:
        report = inspect_extracted_bdd_layout(args.bdd_root)
    except BDDDirectLayoutError as exc:
        report = {
            "status": "HOLD",
            "bdd_root": str(Path(args.bdd_root).expanduser().resolve()),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "claim_boundary": (
                "Adapter input schema is unresolved. No task records, inference records, "
                "model downloads, or training artifacts were created."
            ),
        }
    write_json(args.output, report)
    print(report["status"])
    return 0 if report["status"] == "PASS" else 2


def cmd_bdd_direct_build(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    report = build_direct_bdd_records(
        bdd_root=args.bdd_root,
        output_dir=args.output,
        config=cfg,
        max_train_records=args.max_train_records,
        cross_partition_phash_threshold=args.cross_partition_phash_threshold,
        allow_cross_partition_phash=bool(args.allow_cross_partition_phash),
        phash_compare_backend=args.phash_compare_backend,
        phash_chunk_size=args.phash_chunk_size,
    )
    print(report["status"])
    return 0 if report["status"] == "PASS" else 2


def cmd_local_qlora_feasibility(args: argparse.Namespace) -> int:
    config = LocalQLoRAFeasibilityConfig(
        model_id=args.model_id,
        revision=args.revision,
        image_path=args.image,
        max_image_pixels=args.max_image_pixels,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        memory_budget_fraction=args.memory_budget_fraction,
        run_backward=not bool(args.skip_backward),
    )
    report = run_local_qlora_feasibility(config, args.output)
    print(report["status"])
    return 0 if report["status"] == "PASS" else 2


def cmd_write_config(args: argparse.Namespace) -> int:
    path = write_config_template(args.output, args.template, overwrite=bool(args.force))
    print(path)
    return 0


def cmd_build_data(args: argparse.Namespace) -> int:
    report = build_data_stage(args.annotations, args.images_dir, args.output, args.config)
    print(report["status"])
    return 0 if report["status"] == "PASS" else 2


def cmd_cache_detector(args: argparse.Namespace) -> int:
    report = cache_detector_stage(
        records_path=args.records,
        output=args.output,
        backend=args.backend,
        weights=args.weights,
        config_path=args.config,
        confidence=args.confidence,
        imgsz=args.imgsz,
        device=args.device,
        allow_post_nms_fallback=args.allow_post_nms_fallback,
        detector_miss_policy="record" if args.allow_detector_misses else "fail",
        miss_retry_imgsz=args.miss_retry_imgsz,
        miss_retry_confidence=args.miss_retry_confidence,
    )
    print(report["status"])
    return 0 if report["status"] == "PASS" else 2


def cmd_train_cef(args: argparse.Namespace) -> int:
    report = train_cef_stage(
        args.records,
        args.raw_candidates,
        args.nms_candidates,
        args.output,
        args.config,
        args.detector_misses,
    )
    print(report["status"])
    return 0 if report["status"] == "PASS" else 2


def cmd_prepare_sft(args: argparse.Namespace) -> int:
    report = prepare_sft_dataset_stage(
        records_path=args.records,
        crops_path=args.crops,
        candidate_id=args.candidate_id,
        split=args.split,
        output=args.output,
        max_crops=args.max_crops,
    )
    print(report["status"])
    return 0


def cmd_train_vlm(args: argparse.Namespace) -> int:
    if args.backend == "mock":
        report = rollout_mock_stage(
            args.records, args.crops, args.candidate_id, args.output, args.config
        )
    else:
        if args.execute_training:
            if not args.sft_train:
                raise ValueError("--sft-train is required when --execute-training is set")
            report = qwen_training_stage(args.sft_train, args.sft_dev, args.output, args.config)
        else:
            report = qwen_training_preflight_stage(
                args.output, args.config, args.execute_model_load
            )
    print(report["status"])
    return 0 if report["status"] not in {"MODEL_LOAD_HOLD"} else 2


def cmd_rollout(args: argparse.Namespace) -> int:
    report = rollout_mock_stage(
        args.records, args.crops, args.candidate_id, args.output, args.config
    )
    print(report["status"])
    return 0


def cmd_rollout_qwen(args: argparse.Namespace) -> int:
    report = rollout_qwen_stage(
        records_path=args.records,
        crops_path=args.crops,
        candidate_id=args.candidate_id,
        split=args.split,
        output=args.output,
        adapter_dir=args.adapter_dir,
        config_path=args.config,
    )
    print(report["status"])
    return 0 if report["status"] == "PASS" else 2


def cmd_train_rctr(args: argparse.Namespace) -> int:
    report = train_rctr_stage(
        args.records, args.traces, args.crops, args.output, args.candidate_id, args.config
    )
    print(report["status"])
    return 0 if report["status"] == "PASS" else 2


def cmd_route_review(args: argparse.Namespace) -> int:
    report = route_review_stage(
        records_path=args.records,
        c5_traces_path=args.c5_traces,
        c5_crops_path=args.c5_crops,
        rctr_model_path=args.rctr_model,
        output=args.output,
        config_path=args.config,
    )
    print(report["status"])
    return 0 if report["status"] == "PASS" else 2


def cmd_compare_final(args: argparse.Namespace) -> int:
    report = compare_final_stage(
        records_path=args.records,
        c3_traces_path=args.c3_traces,
        c3_crops_path=args.c3_crops,
        c5_traces_path=args.c5_traces,
        c5_crops_path=args.c5_crops,
        c5_guard_traces_path=args.c5_guard_traces,
        c6_traces_path=args.c6_traces,
        cef_development_gate_path=args.cef_development_gate,
        rctr_routing_report_path=args.rctr_routing_report,
        output=args.output,
        scope=args.scope,
        config_path=args.config,
        c4_traces_path=args.c4_traces,
        c4_crops_path=args.c4_crops,
    )
    print(report["release"]["status"])
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    metrics = evaluate_stage(
        args.records, args.traces, args.crops, args.candidate_id, args.output, args.config
    )
    print(metrics.get("candidate_id"))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from visualact_r.reporting import generate_reports

    root = Path(args.run_dir)
    release = read_json(root / "release_decision.json")
    manifest = read_json(root / "run_manifest.json")
    generate_reports(root, release, manifest)
    print("PASS")
    return 0


def cmd_runtime_provenance(args: argparse.Namespace) -> int:
    result = assert_installed_distribution(args.source_root)
    write_json(args.output, result) if args.output else print(result["package_file"])
    return 0


def cmd_qualify(args: argparse.Namespace) -> int:
    root = Path(args.run_dir)
    manifest = (
        read_json(root / "run_manifest.json") if (root / "run_manifest.json").exists() else {}
    )
    required = [
        "metrics.json",
        "candidate_comparison.csv",
        "paired_comparisons.csv",
        "paired_comparisons.json",
        "slice_metrics.csv",
        "decision_scenarios.csv",
        "counterfactual_evidence.csv",
        "cef_development_gate.json",
        "rctr_reference_library.json",
        "rctr_controlled_stress_test.json",
        "operational_metrics.json",
        "run_manifest.json",
        "report.md",
        "release_decision.json",
    ]
    if not manifest.get("scope", "").startswith("fixture"):
        required.append("rctr_routing_report.json")
    missing = [x for x in required if not (root / x).exists()]
    release = (
        read_json(root / "release_decision.json")
        if (root / "release_decision.json").exists()
        else {"status": "HOLD"}
    )
    coverage_ok = True
    if (root / "candidate_comparison.csv").exists():
        import pandas as pd

        frame = pd.read_csv(root / "candidate_comparison.csv")
        coverage_ok = bool(
            not frame.empty
            and (frame.get("trace_coverage_rate", pd.Series(dtype=float)) >= 1.0).all()
        )
    split_integrity_ok = True
    if not manifest.get("scope", "").startswith("fixture"):
        split_integrity_ok = bool(
            isinstance(manifest.get("split_manifest"), dict)
            and manifest["split_manifest"].get("integrity", {}).get("status") == "PASS"
        )
    routing_contract_ok = True
    if (root / "rctr_routing_report.json").exists():
        routing = read_json(root / "rctr_routing_report.json")
        routing_contract_ok = bool(
            routing.get("status") == "PASS"
            and routing.get("label_fields_read") == 0
            and routing.get("terminal_fields_used_by_rctr") == 0
        )
    status = (
        "PASS"
        if (
            not missing
            and manifest.get("gt_fields_in_inference_contract") == 0
            and coverage_ok
            and split_integrity_ok
            and routing_contract_ok
        )
        else "HOLD"
    )
    report = {
        "status": status,
        "scope": manifest.get("scope"),
        "missing_artifacts": missing,
        "release_status": release.get("status"),
        "trace_coverage_ok": coverage_ok,
        "split_integrity_ok": split_integrity_ok,
        "rctr_routing_contract_ok": routing_contract_ok,
        "claim_boundary": (
            "Fixture PASS validates contracts only; no performance claim."
            if str(manifest.get("scope", "")).startswith("fixture")
            else "Artifact qualification validates protocol integrity; performance wording still requires release gates and saved paired statistics."
        ),
        "run_manifest": manifest,
    }
    write_json(root / "qualification_manifest.json", report)
    print(status)
    return 0 if status == "PASS" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visualact-r", description="VisualAct-R offline research CLI"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("write-config", help="Materialize a bundled editable YAML config template")
    p.add_argument("--template", choices=config_template_names(), default="default")
    p.add_argument("--output", required=True)
    p.add_argument("--force", action="store_true", help="Allow replacing an existing output file")
    p.set_defaults(func=cmd_write_config)

    def config_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", default=None)

    p = sub.add_parser("fixture-run", help="Run synthetic end-to-end contract path")
    config_arg(p)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(func=cmd_fixture)
    p = sub.add_parser("preflight", help="Validate BDD archive/annotation joins")
    p.add_argument("--data-root", required=True)
    p.add_argument("--labels-zip", required=True)
    p.add_argument("--shard", action="append", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_preflight)
    p = sub.add_parser(
        "extract-bdd", help="Extract user-supplied BDD source archives to an external cache"
    )
    p.add_argument("--data-root", required=True)
    p.add_argument("--labels-zip", required=True)
    p.add_argument("--shard", action="append", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing extracted cache only after staging validation",
    )
    p.set_defaults(func=cmd_extract_bdd)
    p = sub.add_parser(
        "bdd-direct-audit",
        help="Read-only audit for an already-extracted BDD train/val layout",
    )
    p.add_argument("--bdd-root", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_bdd_direct_audit)
    p = sub.add_parser(
        "bdd-direct-build",
        help="Build train/dev/calibration from BDD train and frozen final from BDD val",
    )
    config_arg(p)
    p.add_argument("--bdd-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--max-train-records",
        type=int,
        default=None,
        help="Explicit development-only cap. BDD val remains intact and output is not performance-claim ready.",
    )
    p.add_argument("--cross-partition-phash-threshold", type=int, default=4)
    p.add_argument(
        "--phash-compare-backend",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
        help="Backend for cross-boundary pHash Hamming scan. Use cuda to force GPU comparison.",
    )
    p.add_argument(
        "--phash-chunk-size",
        type=int,
        default=256,
        help="Development-record chunk size for CUDA pHash comparison.",
    )
    p.add_argument(
        "--allow-cross-partition-phash",
        action="store_true",
        help="Permit a documented exception after the pHash cross-boundary audit reports matches.",
    )
    p.set_defaults(func=cmd_bdd_direct_build)
    p = sub.add_parser(
        "local-qlora-feasibility",
        help="Run one-image 4-bit NF4 QLoRA forward/backward and write VRAM evidence",
    )
    p.add_argument("--image", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--revision", default="main")
    p.add_argument("--max-image-pixels", type=int, default=262144)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-modules", default="all-linear")
    p.add_argument("--memory-budget-fraction", type=float, default=0.95)
    p.add_argument("--skip-backward", action="store_true")
    p.set_defaults(func=cmd_local_qlora_feasibility)
    p = sub.add_parser(
        "build-data", help="Build task records and leakage-safe splits from extracted images"
    )
    config_arg(p)
    p.add_argument("--annotations", required=True)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_build_data)
    p = sub.add_parser("cache-detector", help="Cache raw pre-NMS and NMS candidates")
    config_arg(p)
    p.add_argument("--records", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--backend", choices=["synthetic", "ultralytics"], required=True)
    p.add_argument("--weights")
    p.add_argument("--confidence", type=float, default=0.10)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device")
    p.add_argument(
        "--allow-post-nms-fallback",
        action="store_true",
        help=(
            "Permit explicitly reported Ultralytics post-NMS fallback candidates when raw "
            "pre-NMS extraction returns no mappable project-class candidates. Use for "
            "detector smoke tests, not raw-pre-NMS CEF claims without a claim-boundary note."
        ),
    )
    p.add_argument(
        "--allow-detector-misses",
        action="store_true",
        help=(
            "Record images with no mappable detector candidates to detector_misses.jsonl "
            "instead of failing the cache stage. The report keeps miss count and coverage "
            "metrics so detector misses remain explicit evidence."
        ),
    )
    p.add_argument(
        "--miss-retry-imgsz",
        type=int,
        help=(
            "Optional second-pass image size used only for images that have no mappable "
            "raw candidates at the primary detector setting."
        ),
    )
    p.add_argument(
        "--miss-retry-confidence",
        type=float,
        help=(
            "Optional second-pass confidence used with --miss-retry-imgsz. Defaults to "
            "the primary --confidence when omitted."
        ),
    )
    p.set_defaults(func=cmd_cache_detector)
    p = sub.add_parser("train-cef", help="Train CEF directional corrector and crop ranker")
    config_arg(p)
    p.add_argument("--records", required=True)
    p.add_argument("--raw-candidates", required=True)
    p.add_argument("--nms-candidates", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--detector-misses",
        help="Optional detector_misses.jsonl produced by cache-detector --allow-detector-misses",
    )
    p.set_defaults(func=cmd_train_cef)
    p = sub.add_parser(
        "prepare-sft", help="Build train/dev-only local QLoRA JSONL and derived evidence crops"
    )
    p.add_argument("--records", required=True)
    p.add_argument("--crops")
    p.add_argument("--candidate-id", choices=["C2", "C3", "C4", "C5"], required=True)
    p.add_argument("--split", choices=["train", "dev"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-crops", type=int, default=3)
    p.set_defaults(func=cmd_prepare_sft)
    p = sub.add_parser(
        "train-vlm", help="Run mock traces, QLoRA preflight, or a local QLoRA SFT job"
    )
    config_arg(p)
    p.add_argument("--backend", choices=["mock", "qwen"], required=True)
    p.add_argument("--records")
    p.add_argument("--crops")
    p.add_argument("--candidate-id", default="C5")
    p.add_argument("--output", required=True)
    p.add_argument("--sft-train")
    p.add_argument("--sft-dev")
    p.add_argument("--execute-model-load", action="store_true")
    p.add_argument("--execute-training", action="store_true")
    p.set_defaults(func=cmd_train_vlm)
    p = sub.add_parser("rollout", help="Produce typed mock agent traces")
    config_arg(p)
    p.add_argument("--records", required=True)
    p.add_argument("--crops", required=True)
    p.add_argument("--candidate-id", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_rollout)
    p = sub.add_parser(
        "rollout-qwen",
        help="Run bounded local Qwen visual-agent traces on one sealed split",
    )
    config_arg(p)
    p.add_argument("--records", required=True)
    p.add_argument("--crops")
    p.add_argument("--candidate-id", choices=["C2", "C3", "C4", "C5"], required=True)
    p.add_argument("--split", choices=["train", "dev", "calibration", "final"], required=True)
    p.add_argument("--adapter-dir")
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_rollout_qwen)
    p = sub.add_parser("train-rctr", help="Fit train-only reference-conditioned critic")
    config_arg(p)
    p.add_argument("--records", required=True)
    p.add_argument("--traces", required=True)
    p.add_argument("--crops", required=True)
    p.add_argument("--candidate-id", default="C5")
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_train_rctr)
    p = sub.add_parser(
        "route-review",
        help="Create fixed-capacity confidence and RCTR review routes from sealed C5 traces",
    )
    config_arg(p)
    p.add_argument("--records", required=True)
    p.add_argument("--c5-traces", required=True)
    p.add_argument("--c5-crops", required=True)
    p.add_argument("--rctr-model", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_route_review)
    p = sub.add_parser(
        "compare-final",
        help="Create the canonical C3/C5/C5-confidence/C6 frozen-final evidence package",
    )
    config_arg(p)
    p.add_argument("--records", required=True)
    p.add_argument("--c3-traces", required=True)
    p.add_argument("--c3-crops", required=True)
    p.add_argument("--c5-traces", required=True)
    p.add_argument("--c5-crops", required=True)
    p.add_argument("--c4-traces")
    p.add_argument("--c4-crops")
    p.add_argument("--c5-guard-traces", required=True)
    p.add_argument("--c6-traces", required=True)
    p.add_argument("--cef-development-gate", required=True)
    p.add_argument("--rctr-routing-report", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--scope",
        choices=["bdd100k_derived_internal_frozen_final", "fixture_only"],
        default="bdd100k_derived_internal_frozen_final",
    )
    p.set_defaults(func=cmd_compare_final)
    p = sub.add_parser("evaluate", help="Evaluate one candidate on final split")
    config_arg(p)
    p.add_argument("--records", required=True)
    p.add_argument("--traces", required=True)
    p.add_argument("--crops", required=True)
    p.add_argument("--candidate-id", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_evaluate)
    p = sub.add_parser("report", help="Regenerate report from run artifacts")
    p.add_argument("--run-dir", required=True)
    p.set_defaults(func=cmd_report)
    p = sub.add_parser(
        "runtime-provenance",
        help="Fail if a wheel/sdist smoke test imports the checkout source tree",
    )
    p.add_argument("--source-root", required=True)
    p.add_argument("--output")
    p.set_defaults(func=cmd_runtime_provenance)
    p = sub.add_parser("qualify", help="Validate artifact contract and claim boundary")
    p.add_argument("--run-dir", required=True)
    p.set_defaults(func=cmd_qualify)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        exit_code = int(args.func(args))
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        # Expected operational failures should be actionable and should not leave the user
        # with a long implementation traceback. Set VISUALACT_R_DEBUG=1 for developer traces.
        if os.environ.get("VISUALACT_R_DEBUG") == "1":
            raise
        print(f"ERROR [{args.command}]: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    raise SystemExit(exit_code)
