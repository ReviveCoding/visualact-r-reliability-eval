from dataclasses import replace

import numpy as np

from visualact_r.rctr.core import apply_review_routes, prefix_features, select_review_routes
from visualact_r.schemas import AgentStep, AgentTrace


def _trace(image_id: str, action: str = "AUTO_CLEAR", confidence: float = 0.8) -> AgentTrace:
    return AgentTrace(
        trace_id=f"{image_id}:C5",
        image_id=image_id,
        task_id="test",
        candidate_id="C5",
        visible_crop_ids=["crop_a"],
        steps=[
            AgentStep(1, "INSPECT_REGION", "crop_a", None, 0.55, True, "inspect", 0.0),
            AgentStep(2, "FINALIZE", None, action, confidence, True, "finalize", 0.0),
        ],
        final_decision=action,  # type: ignore[arg-type]
        final_confidence=confidence,
        schema_valid=True,
        latency_ms=0.0,
    )


def test_prefix_features_ignore_terminal_decision_and_confidence():
    source = _trace("case")
    altered = replace(source, final_decision="ESCALATE", final_confidence=0.01)
    assert np.array_equal(
        prefix_features(source, [], prefix_len=1), prefix_features(altered, [], prefix_len=1)
    )


def test_fixed_capacity_routing_uses_only_auto_clear_candidates():
    traces = [_trace("a"), _trace("b"), _trace("c"), _trace("d", action="REVIEW"), _trace("e")]
    routed = select_review_routes(
        traces, {"a": 0.1, "b": 0.9, "c": 0.8, "d": 1.0, "e": 0.2}, review_capacity=0.4
    )
    assert routed == {"b", "c"}
    routed_traces = apply_review_routes(traces, routed, {"b": 0.9, "c": 0.8}, {}, candidate_id="C6")
    for trace in routed_traces:
        if trace.image_id in routed:
            assert trace.review_routed
            assert trace.pre_route_decision == "AUTO_CLEAR"
            assert all(step.typed_action != "FINALIZE" for step in trace.steps)
            assert trace.steps[-1].typed_action == "ROUTE_REVIEW"
