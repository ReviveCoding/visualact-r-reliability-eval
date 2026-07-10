from pathlib import Path

import pytest

from visualact_r.agents.core import parse_typed_output
from visualact_r.artifacts import load_inference_records
from visualact_r.config import load_config
from visualact_r.evaluation.metrics import evaluate_candidate
from visualact_r.schemas import (
    AgentStep,
    AgentTrace,
    Box,
    CropRecord,
    GroundTruthObject,
    TaskRecord,
    rectangle_union_area,
)
from visualact_r.utils import write_jsonl


def test_packaged_config_loads_outside_repository(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert cfg["project"]["name"] == "visualact-r"


def test_strict_typed_parser_handles_only_complete_json() -> None:
    parsed = parse_typed_output(
        '```json\n{"tool":"FINALIZE","decision":"REVIEW","confidence":0.5,"evidence_ids":["a"],"reason_code":"test"}\n```'
    )
    assert parsed["decision"] == "REVIEW"
    with pytest.raises(ValueError):
        parse_typed_output(
            '{"tool":"FINALIZE","decision":"REVIEW","confidence":0.5,"evidence_ids":[],"reason_code":"x"} extra'
        )
    with pytest.raises(ValueError):
        parse_typed_output(
            '{"tool":"FINALIZE","decision":"REVIEW","confidence":0.5,"evidence_ids":["a","a"],"reason_code":"x"}'
        )


def test_union_area_does_not_double_count_overlap() -> None:
    assert rectangle_union_area([Box(0, 0, 2, 2), Box(1, 0, 3, 2)]) == 6.0


def test_grounded_metric_uses_final_evidence_not_all_visible_crops() -> None:
    record = TaskRecord(
        image_id="i",
        image_path="unused.png",
        width=100,
        height=100,
        task_id="task",
        task_template_id="template",
        prompt="p",
        weather="clear",
        scene="city",
        timeofday="daytime",
        gt_objects=[GroundTruthObject("support", "person", Box(0, 0, 10, 10))],
        action_label="REVIEW",
        supporting_object_ids=["support"],
        split="final",
    )
    crop = CropRecord(
        crop_id="support_crop",
        image_id="i",
        query_id="q",
        box_xyxy=Box(0, 0, 10, 10),
        primary_candidate_id="raw",
        raw_candidate_ids=["raw"],
        category="person",
        base_confidence=0.9,
        edge_expansion=[0.0] * 4,
        edge_deflation=[0.0] * 4,
        coverage_proxy=0.9,
        contamination_proxy=0.0,
        selection_rank=1,
        variant="cef",
    )
    trace = AgentTrace(
        trace_id="t",
        image_id="i",
        task_id="task",
        candidate_id="C5",
        visible_crop_ids=["support_crop"],
        final_evidence_ids=[],
        steps=[AgentStep(1, "FINALIZE", None, "REVIEW", 0.9, True, "x", 0.0)],
        final_decision="REVIEW",
        final_confidence=0.9,
        schema_valid=True,
        latency_ms=0.0,
    )
    _, rows = evaluate_candidate([record], [trace], {"i": [crop]}, "C5")
    assert rows[0]["available_critical_evidence_recall"] == 1.0
    assert rows[0]["critical_evidence_recall"] == 0.0
    assert rows[0]["grounded_action_success"] == 0


def test_inference_record_excludes_label_fields(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    write_jsonl(
        path,
        [
            {
                "image_id": "i",
                "image_path": "x.png",
                "width": 1,
                "height": 1,
                "task_id": "t",
                "task_template_id": "v",
                "prompt": "p",
                "split": "final",
                "label_policy_version": "v1",
                "action_label": "ESCALATE",
                "supporting_object_ids": ["secret"],
                "gt_objects": [],
            }
        ],
    )
    with pytest.raises(ValueError, match="forbidden non-serving fields"):
        load_inference_records(path, "final")

    safe_path = tmp_path / "inference_records.jsonl"
    write_jsonl(
        safe_path,
        [
            {
                "image_id": "i",
                "image_path": "x.png",
                "width": 1,
                "height": 1,
                "task_id": "t",
                "task_template_id": "v",
                "prompt": "p",
                "split": "final",
                "label_policy_version": "v1",
            }
        ],
    )
    rows = load_inference_records(safe_path, "final")
    assert rows[0].image_id == "i"
    assert not hasattr(rows[0], "action_label")
