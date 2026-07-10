from pathlib import Path

from visualact_r.pipeline import run_fixture_pipeline
from visualact_r.stages import compare_final_stage, route_review_stage


def test_sealed_final_comparison_contract(tmp_path: Path):
    fixture = tmp_path / "fixture"
    run_fixture_pipeline(fixture)
    routing = tmp_path / "routing"
    routing_report = route_review_stage(
        records_path=str(fixture / "data" / "task_records.jsonl"),
        c5_traces_path=str(fixture / "artifacts" / "agent_traces.jsonl"),
        c5_crops_path=str(fixture / "artifacts" / "crop_records.jsonl"),
        rctr_model_path=str(fixture / "artifacts" / "rctr_model.pkl"),
        output=str(routing),
    )
    assert routing_report["status"] == "PASS"
    assert routing_report["label_fields_read"] == 0
    final = tmp_path / "final"
    report = compare_final_stage(
        records_path=str(fixture / "data" / "task_records.jsonl"),
        c3_traces_path=str(fixture / "artifacts" / "agent_traces.jsonl"),
        c3_crops_path=str(fixture / "artifacts" / "crop_records.jsonl"),
        c5_traces_path=str(fixture / "artifacts" / "agent_traces.jsonl"),
        c5_crops_path=str(fixture / "artifacts" / "crop_records.jsonl"),
        c5_guard_traces_path=str(routing / "agent_traces_C5_CONFIDENCE_GUARD.jsonl"),
        c6_traces_path=str(routing / "agent_traces_C6.jsonl"),
        cef_development_gate_path=str(fixture / "cef_development_gate.json"),
        rctr_routing_report_path=str(routing / "rctr_routing_report.json"),
        output=str(final),
        scope="fixture_only",
    )
    assert report["status"] == "PASS"
    assert report["release"]["status"] == "FIXTURE_ONLY_NO_PERFORMANCE_CLAIM"
    assert (final / "rctr_routing_report.json").exists()
    assert (final / "paired_comparisons.json").exists()
