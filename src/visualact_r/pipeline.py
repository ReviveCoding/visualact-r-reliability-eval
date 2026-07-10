from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

from visualact_r.agents.core import MockVisualAgent
from visualact_r.baselines import CalibratedDetectorTriage, c0_policy
from visualact_r.cef.core import evaluate_development_gate, fit_cef, make_crops
from visualact_r.config import load_config
from visualact_r.data.build import build_task_records
from visualact_r.data.fixtures import create_synthetic_fixture
from visualact_r.data.preflight import preflight_fixture
from visualact_r.data.splits import build_splits, validate_split_integrity
from visualact_r.detector.synthetic import SyntheticColorDetector
from visualact_r.evaluation.metrics import (
    evaluate_candidate,
    paired_bootstrap_delta,
    review_capacity_scenarios,
    slice_metrics,
)
from visualact_r.evaluation.release import release_decision
from visualact_r.rctr.core import apply_review_routes, fit_rctr, select_review_routes
from visualact_r.reporting import generate_reports
from visualact_r.stages import write_project_creation_handoff
from visualact_r.schemas import AgentStep, AgentTrace, CropRecord, TaskRecord, intersection_box
from visualact_r.utils import (
    ensure_dir,
    seed_everything,
    sha256_file,
    stable_hash,
    utc_now,
    write_json,
    write_jsonl,
    write_table,
)


def _subset(records: list[TaskRecord], split: str) -> list[TaskRecord]:
    return [x for x in records if x.split == split]


def _serialize_cache(cache: dict[str, tuple[list[Any], list[Any]]], root: Path) -> None:
    raw_rows = []
    nms_rows = []
    for raw, nms in cache.values():
        raw_rows.extend([x.to_dict() for x in raw])
        nms_rows.extend([x.to_dict() for x in nms])
    write_jsonl(root / "detector_raw_candidates.jsonl", raw_rows)
    write_jsonl(root / "detector_nms_candidates.jsonl", nms_rows)


def _trace_from_policy(
    record: TaskRecord, nms: list[Any], config: dict[str, Any], candidate_id: str
) -> AgentTrace:
    action, confidence = c0_policy(record, nms, config)
    step = AgentStep(
        step=1,
        typed_action="FINALIZE",
        region_id=None,
        decision=action,
        confidence=confidence,
        schema_valid=True,
        reason_code="detector_zone_policy",
        latency_ms=0.0,
    )
    return AgentTrace(
        trace_id=f"{record.image_id}:{candidate_id}",
        image_id=record.image_id,
        task_id=record.task_id,
        candidate_id=candidate_id,
        visible_crop_ids=[],
        steps=[step],
        final_decision=action,
        final_confidence=confidence,
        schema_valid=True,
        latency_ms=0.0,
    )


def _trace_from_c1(
    record: TaskRecord, nms: list[Any], model: CalibratedDetectorTriage, config: dict[str, Any]
) -> AgentTrace:
    action, confidence, _ = model.predict(record, nms, config)
    step = AgentStep(
        step=1,
        typed_action="FINALIZE",
        region_id=None,
        decision=action,
        confidence=confidence,
        schema_valid=True,
        reason_code="calibrated_detector_triage",
        latency_ms=0.0,
    )
    return AgentTrace(
        trace_id=f"{record.image_id}:C1",
        image_id=record.image_id,
        task_id=record.task_id,
        candidate_id="C1",
        visible_crop_ids=[],
        steps=[step],
        final_decision=action,
        final_confidence=confidence,
        schema_valid=True,
        latency_ms=0.0,
    )


def _counterfactual_rows(
    records: list[TaskRecord], crops_by_image: dict[str, list[CropRecord]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Fixture-only crop-removal contract test, never VLM causal evidence."""
    agent = MockVisualAgent("C5_COUNTERFACTUAL")
    rows: list[dict[str, Any]] = []
    for record in records:
        crops = crops_by_image.get(record.image_id, [])
        support = [
            obj for obj in record.gt_objects if obj.object_id in record.supporting_object_ids
        ]
        if not crops or not support:
            continue
        overlap: dict[str, float] = {}
        for crop in crops:
            area = 0.0
            for obj in support:
                intersection = intersection_box(crop.box_xyxy, obj.box_xyxy)
                if intersection is not None:
                    area += intersection.area
            overlap[crop.crop_id] = area
        critical = max(crops, key=lambda crop: overlap[crop.crop_id])
        irrelevant = min(crops, key=lambda crop: overlap[crop.crop_id])
        baseline = agent.rollout(record, crops, config)
        removed_critical = agent.rollout(
            record, [crop for crop in crops if crop.crop_id != critical.crop_id], config
        )
        removed_irrelevant = agent.rollout(
            record, [crop for crop in crops if crop.crop_id != irrelevant.crop_id], config
        )
        rows.append(
            {
                "image_id": record.image_id,
                "scope": "fixture_mock_only",
                "baseline_decision": baseline.final_decision,
                "critical_crop_id": critical.crop_id,
                "irrelevant_crop_id": irrelevant.crop_id,
                "decision_after_critical_removal": removed_critical.final_decision,
                "decision_after_irrelevant_removal": removed_irrelevant.final_decision,
                "critical_removal_changed_decision": int(
                    baseline.final_decision != removed_critical.final_decision
                ),
                "irrelevant_removal_changed_decision": int(
                    baseline.final_decision != removed_irrelevant.final_decision
                ),
            }
        )
    return rows


def _rctr_controlled_stress_test() -> dict[str, Any]:
    """Exercise capacity-matched RCTR routing without presenting synthetic traces as model evidence."""
    traces: list[AgentTrace] = []
    expected_unsafe: dict[str, bool] = {}
    risks: dict[str, float] = {}
    for idx, (risk, unsafe) in enumerate(
        [(0.97, True), (0.83, True), (0.41, False), (0.15, False), (0.09, False)]
    ):
        image_id = f"controlled_stress_{idx}"
        traces.append(
            AgentTrace(
                trace_id=f"{image_id}:C5",
                image_id=image_id,
                task_id="controlled_stress",
                candidate_id="C5",
                visible_crop_ids=[f"crop_{idx}"],
                steps=[
                    AgentStep(
                        step=1,
                        typed_action="INSPECT_REGION",
                        region_id=f"crop_{idx}",
                        decision=None,
                        confidence=0.55,
                        schema_valid=True,
                        reason_code="controlled_prefix",
                        latency_ms=0.0,
                    ),
                    AgentStep(
                        step=2,
                        typed_action="FINALIZE",
                        region_id=None,
                        decision="AUTO_CLEAR",
                        confidence=0.80,
                        schema_valid=True,
                        reason_code="controlled_terminal",
                        latency_ms=0.0,
                    ),
                ],
                final_decision="AUTO_CLEAR",
                final_confidence=0.80,
                schema_valid=True,
                latency_ms=0.0,
            )
        )
        risks[image_id] = risk
        expected_unsafe[image_id] = unsafe
    routed = select_review_routes(traces, risks, review_capacity=0.40)
    routed_traces = apply_review_routes(
        traces,
        routed,
        risks,
        {trace.image_id: ["controlled_reference@1"] for trace in traces},
        candidate_id="C6_CONTROLLED_STRESS",
        reason_code="controlled_prefix_risk_route",
    )
    unsafe_routed = sum(expected_unsafe[image_id] for image_id in routed)
    terminal_removed = all(
        any(step.typed_action == "ROUTE_REVIEW" for step in trace.steps)
        and all(step.typed_action != "FINALIZE" for step in trace.steps)
        for trace in routed_traces
        if trace.review_routed
    )
    return {
        "scope": "controlled_synthetic_contract_not_model_evidence",
        "review_capacity": 0.40,
        "eligible_auto_clear_count": len(traces),
        "routed_count": len(routed),
        "routed_ids": sorted(routed),
        "expected_top_risk_ids": ["controlled_stress_0", "controlled_stress_1"],
        "unsafe_capture_recall": unsafe_routed / max(1, sum(expected_unsafe.values())),
        "route_replaces_terminal_finalize": terminal_removed,
        "status": "PASS"
        if sorted(routed) == ["controlled_stress_0", "controlled_stress_1"] and terminal_removed
        else "HOLD",
    }


def run_fixture_pipeline(
    output_dir: str | Path, config_path: str | Path | None = None, seed: int | None = None
) -> dict[str, Any]:
    root = ensure_dir(output_dir)
    config = load_config(config_path)
    if seed is not None:
        config["project"]["seed"] = int(seed)
    seed_everything(int(config["project"]["seed"]))
    fixture_root = ensure_dir(root / "fixture_source")
    source = create_synthetic_fixture(fixture_root)
    preflight = preflight_fixture(fixture_root, root)
    records = build_task_records(source["annotations"], source["images_dir"], config, root / "data")
    records = build_splits(records, config, str(root / "data"))
    integrity = validate_split_integrity(records, config)
    detector = SyntheticColorDetector()
    cache = {record.image_id: detector.predict(record) for record in records}
    _serialize_cache(cache, root / "artifacts")
    train = _subset(records, "train")
    dev = _subset(records, "dev")
    final = _subset(records, "final")
    c1 = CalibratedDetectorTriage().fit(
        train, {x.image_id: cache[x.image_id][1] for x in train}, config
    )
    c1.save(root / "artifacts" / "c1_detector_triage.pkl")
    cef = fit_cef(train, cache, config)
    cef_gate = evaluate_development_gate(cef, dev, cache, config)
    cef.development_gate = cef_gate
    cef.save(root / "artifacts" / "cef_model.pkl")
    write_json(root / "cef_development_gate.json", cef_gate)
    all_crops: dict[str, dict[str, list[CropRecord]]] = {"C2": {}, "C3": {}, "C4": {}, "C5": {}}
    for record in records:
        raw, nms = cache[record.image_id]
        all_crops["C2"][record.image_id] = []
        all_crops["C3"][record.image_id] = make_crops(record, raw, nms, "nms", config)
        all_crops["C4"][record.image_id] = make_crops(record, raw, nms, "raw_ranker", config, cef)
        all_crops["C5"][record.image_id] = make_crops(record, raw, nms, "cef", config, cef)
    crop_rows = []
    for candidate, mapping in all_crops.items():
        for crop_list in mapping.values():
            crop_rows.extend([{**x.to_dict(), "candidate_id": candidate} for x in crop_list])
    write_jsonl(root / "artifacts" / "crop_records.jsonl", crop_rows)
    traces: dict[str, list[AgentTrace]] = {key: [] for key in ["C0", "C1", "C2", "C3", "C4", "C5"]}
    for record in records:
        raw, nms = cache[record.image_id]
        traces["C0"].append(_trace_from_policy(record, nms, config, "C0"))
        traces["C1"].append(_trace_from_c1(record, nms, c1, config))
        for candidate in ["C2", "C3", "C4", "C5"]:
            traces[candidate].append(
                MockVisualAgent(candidate).rollout(
                    record, all_crops[candidate][record.image_id], config
                )
            )
    rctr = fit_rctr(
        train,
        [
            trace
            for trace in traces["C5"]
            if trace.image_id in {record.image_id for record in train}
        ],
        all_crops["C5"],
        n_reference=int(config["rctr"]["max_reference_library"]),
        neighbor_count=int(config["rctr"]["n_reference"]),
    )
    rctr_model_path = root / "artifacts" / "rctr_model.pkl"
    rctr.save(rctr_model_path)
    write_json(
        root / "artifacts" / "rctr_training_report.json",
        {
            "status": "PASS",
            "references_train_only": True,
            "prefix_only_contract": True,
            "train_trace_count": sum(
                trace.image_id in {record.image_id for record in train} for trace in traces["C5"]
            ),
            "rctr_model_sha256": sha256_file(rctr_model_path),
            "scope": "fixture_contract_only",
        },
    )
    rctr_risk: dict[str, float] = {}
    rctr_refs: dict[str, list[str]] = {}
    for trace in traces["C5"]:
        rctr_value, refs = rctr.risk(
            trace,
            all_crops["C5"].get(trace.image_id, []),
            int(config["rctr"]["n_reference"]),
            prefix_len=max(0, len(trace.steps) - 1),
        )
        rctr_risk[trace.image_id] = rctr_value
        rctr_refs[trace.image_id] = refs
    write_json(
        root / "rctr_reference_library.json",
        {
            "reference_prefix_count": len(rctr.reference_prefix_ids),
            "reference_trace_count": len(rctr.reference_trace_ids),
            "feature_names": rctr.feature_names,
            "references_train_only": True,
        },
    )
    rctr_controlled_stress = _rctr_controlled_stress_test()
    write_json(root / "rctr_controlled_stress_test.json", rctr_controlled_stress)
    review_capacity = float(config["rctr"]["review_capacity"])
    c5_confidence_risk = {trace.image_id: 1.0 - trace.final_confidence for trace in traces["C5"]}
    c5g_routes = select_review_routes(traces["C5"], c5_confidence_risk, review_capacity)
    c6_routes = select_review_routes(traces["C5"], rctr_risk, review_capacity)
    traces["C5_CONFIDENCE_GUARD"] = apply_review_routes(
        traces["C5"],
        c5g_routes,
        c5_confidence_risk,
        {},
        candidate_id="C5_CONFIDENCE_GUARD",
        reason_code="confidence_only_capacity_route",
    )
    traces["C6"] = apply_review_routes(
        traces["C5"],
        c6_routes,
        rctr_risk,
        rctr_refs,
        candidate_id="C6",
        reason_code="rctr_prefix_risk_capacity_route",
    )
    write_jsonl(
        root / "artifacts" / "agent_traces.jsonl",
        [trace.to_dict() for candidate_traces in traces.values() for trace in candidate_traces],
    )
    metrics_by_candidate: dict[str, dict[str, Any]] = {}
    candidate_rows: dict[str, list[dict[str, Any]]] = {}
    comparison_rows: list[dict[str, Any]] = []
    all_slice_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    final_image_ids = {record.image_id for record in final}
    for candidate, candidate_traces in traces.items():
        if candidate in {"C5_CONFIDENCE_GUARD", "C6"}:
            candidate_crops = all_crops["C5"]
        else:
            candidate_crops = all_crops.get(candidate, {})
        metric, rows = evaluate_candidate(
            final,
            [trace for trace in candidate_traces if trace.image_id in final_image_ids],
            candidate_crops,
            candidate,
            config,
        )
        metrics_by_candidate[candidate] = metric
        candidate_rows[candidate] = rows
        comparison_rows.append(metric)
        all_slice_rows.extend([{**row, "candidate_id": candidate} for row in slice_metrics(rows)])
        risk: dict[str, float]
        if candidate == "C6":
            risk = rctr_risk
        elif candidate == "C5_CONFIDENCE_GUARD":
            risk = c5_confidence_risk
        else:
            risk = {trace.image_id: 1.0 - trace.final_confidence for trace in candidate_traces}
        scenario_rows.extend(
            [{**row, "candidate_id": candidate} for row in review_capacity_scenarios(rows, risk)]
        )
    comparisons = {
        "C5_minus_C3_hard_grounded_action_success": paired_bootstrap_delta(
            candidate_rows["C3"],
            candidate_rows["C5"],
            "grounded_action_success",
            final_group="hard",
            iterations=int(config["evaluation"]["bootstrap_iterations"]),
            seed=int(config["project"]["seed"]),
            min_samples=int(config["evaluation"]["min_paired_final_samples"]),
        ),
        "C5_minus_C3_all_critical_evidence_recall": paired_bootstrap_delta(
            candidate_rows["C3"],
            candidate_rows["C5"],
            "critical_evidence_recall",
            iterations=int(config["evaluation"]["bootstrap_iterations"]),
            seed=int(config["project"]["seed"]),
            min_samples=int(config["evaluation"]["min_paired_final_samples"]),
        ),
        "C6_minus_C5_confidence_guard_grounded_action_success": paired_bootstrap_delta(
            candidate_rows["C5_CONFIDENCE_GUARD"],
            candidate_rows["C6"],
            "grounded_action_success",
            iterations=int(config["evaluation"]["bootstrap_iterations"]),
            seed=int(config["project"]["seed"]),
            min_samples=int(config["evaluation"]["min_paired_final_samples"]),
        ),
    }
    comparison_table = [{"comparison_id": name, **value} for name, value in comparisons.items()]
    write_json(root / "metrics.json", metrics_by_candidate)
    write_json(root / "paired_comparisons.json", comparisons)
    write_table(
        comparison_rows, root / "candidate_comparison.csv", root / "candidate_comparison.parquet"
    )
    write_table(
        comparison_table, root / "paired_comparisons.csv", root / "paired_comparisons.parquet"
    )
    write_table(all_slice_rows, root / "slice_metrics.csv", root / "slice_metrics.parquet")
    write_table(scenario_rows, root / "decision_scenarios.csv", root / "decision_scenarios.parquet")
    counterfactual = _counterfactual_rows(final, all_crops["C5"], config)
    write_table(
        counterfactual,
        root / "counterfactual_evidence.csv",
        root / "counterfactual_evidence.parquet",
    )
    operational = {
        "scope": "fixture_only",
        "gpu_peak_memory_mb": None,
        "model_artifact_count": 3,
        "detector_backend": detector.source,
        "raw_candidate_extraction_validated": True,
        "cef_development_gate": cef_gate,
        "rctr_prefix_only": True,
        "rctr_reference_prefix_count": len(rctr.reference_prefix_ids),
        "rctr_controlled_stress_status": rctr_controlled_stress["status"],
        "review_capacity_policy": review_capacity,
        "hardware": platform.platform(),
    }
    write_json(root / "operational_metrics.json", operational)
    manifest = {
        "run_id": stable_hash(
            {"root": str(root), "seed": config["project"]["seed"], "created": utc_now()}
        )[:16],
        "scope": "fixture_only",
        "created_at_utc": utc_now(),
        "config_hash": stable_hash(config),
        "dataset_fingerprint": stable_hash([x.image_id for x in records]),
        "seed": config["project"]["seed"],
        "preflight_status": preflight["status"],
        "split_integrity": integrity,
        "gt_fields_in_inference_contract": 0,
        "candidate_ids": list(traces.keys()),
        "rctr_prefix_feature_contract": "pre-final trace steps only; terminal action/confidence excluded",
    }
    write_json(root / "run_manifest.json", manifest)
    release = release_decision(metrics_by_candidate, "fixture_only", config, comparisons)
    generate_reports(root, release, manifest)
    write_project_creation_handoff(root, config, preflight["status"])
    return {"run_dir": str(root), "preflight": preflight, "release": release, "manifest": manifest}
