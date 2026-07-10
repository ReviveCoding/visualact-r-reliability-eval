from __future__ import annotations

from typing import Any


def _pp(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return 100.0 * (right - left)


def release_decision(
    metrics_by_candidate: dict[str, dict[str, Any]],
    scope: str,
    config: dict[str, Any],
    comparisons: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if scope.startswith("fixture"):
        return {
            "status": "FIXTURE_ONLY_NO_PERFORMANCE_CLAIM",
            "reason": "Synthetic fixture runs validate contracts and artifacts, not model performance claims.",
            "gates": [],
        }
    c3 = metrics_by_candidate.get("C3")
    c5 = metrics_by_candidate.get("C5")
    c5g = metrics_by_candidate.get("C5_CONFIDENCE_GUARD")
    c6 = metrics_by_candidate.get("C6")
    if not c3 or not c5:
        return {"status": "HOLD", "reason": "Required C3/C5 comparison missing", "gates": []}
    cfg = config["release"]
    required_integrity = {"C3": c3, "C5": c5}
    if c5g is not None:
        required_integrity["C5_CONFIDENCE_GUARD"] = c5g
    if c6 is not None:
        required_integrity["C6"] = c6
    integrity_failures = {
        candidate: {
            "trace_coverage_rate": metrics.get("trace_coverage_rate"),
            "schema_failure_rate": metrics.get("schema_failure_rate"),
        }
        for candidate, metrics in required_integrity.items()
        if float(metrics.get("trace_coverage_rate", 0.0)) < float(cfg["require_trace_coverage"])
        or float(metrics.get("schema_failure_rate", 1.0)) > float(cfg["max_schema_failure_rate"])
    }
    if integrity_failures:
        return {
            "status": "HOLD",
            "reason": "Required candidate trace/schema integrity gate failed",
            "gates": {"operational_integrity": integrity_failures},
        }
    hard_gas_gain = _pp(
        c3.get("hard_grounded_action_success"), c5.get("hard_grounded_action_success")
    )
    recall_gain = _pp(c3.get("critical_evidence_recall"), c5.get("critical_evidence_recall"))
    contamination_delta = _pp(c3.get("evidence_contamination"), c5.get("evidence_contamination"))
    clean_gas_delta = _pp(
        c3.get("clean_grounded_action_success"), c5.get("clean_grounded_action_success")
    )
    ci = (comparisons or {}).get("C5_minus_C3_hard_grounded_action_success", {})
    trace_gate = float(c5.get("trace_coverage_rate", 0.0)) >= float(cfg["require_trace_coverage"])
    c5_gates = {
        "hard_gas_gain_pp": hard_gas_gain,
        "critical_evidence_recall_gain_pp": recall_gain,
        "contamination_delta_pp": contamination_delta,
        "clean_gas_delta_pp": clean_gas_delta,
        "false_auto_clear_delta": c5.get("false_auto_clear_rate", 1.0)
        - c3.get("false_auto_clear_rate", 1.0),
        "paired_hard_gas_ci95_low": ci.get("ci95_low"),
        "trace_coverage_gate": trace_gate,
    }
    passed_c5 = (
        hard_gas_gain is not None
        and hard_gas_gain >= float(cfg["c5_gas_gain_pp"])
        and recall_gain is not None
        and recall_gain >= float(cfg["c5_critical_recall_gain_pp"])
        and contamination_delta is not None
        and contamination_delta <= float(cfg["c5_contamination_tolerance_pp"])
        and clean_gas_delta is not None
        and clean_gas_delta >= -float(cfg["c5_clean_regression_pp"])
        and c5_gates["false_auto_clear_delta"] <= 0.0
        and ci.get("status") == "PASS"
        and ci.get("ci95_low") is not None
        and float(ci["ci95_low"]) > 0.0
        and trace_gate
        and float(c5.get("schema_failure_rate", 1.0)) <= float(cfg["max_schema_failure_rate"])
    )
    if not passed_c5:
        return {
            "status": "ITERATE",
            "reason": "CEF evidence-upgrade gates not met on frozen final protocol",
            "gates": {"C5": c5_gates},
        }
    if c5g is None or c6 is None:
        return {
            "status": "EVIDENCE_UPGRADE_C5_ONLY",
            "reason": "C5 meets the primary frozen evidence gate; matched-capacity C6 evidence unavailable.",
            "gates": {"C5": c5_gates},
        }
    baseline_false = float(c5g.get("false_auto_clear_rate", 0.0))
    c6_false = float(c6.get("false_auto_clear_rate", 0.0))
    false_reduction = (baseline_false - c6_false) / baseline_false if baseline_false > 0.0 else None
    unsafe_gain = _pp(c5g.get("routed_unsafe_capture_rate"), c6.get("routed_unsafe_capture_rate"))
    clean_macro_delta = _pp(c5g.get("clean_macro_f1"), c6.get("clean_macro_f1"))
    matched_capacity = abs(
        float(c5g.get("review_routed_rate", 0.0)) - float(c6.get("review_routed_rate", 0.0))
    ) <= (1.0 / max(1, int(c6.get("n", 1))))
    c6_gates = {
        "matched_review_capacity": matched_capacity,
        "confidence_guard_review_routed_rate": c5g.get("review_routed_rate"),
        "rctr_review_routed_rate": c6.get("review_routed_rate"),
        "false_auto_clear_relative_reduction": false_reduction,
        "unsafe_capture_gain_pp": unsafe_gain,
        "clean_macro_f1_delta_pp": clean_macro_delta,
        "schema_failure_rate": c6.get("schema_failure_rate"),
        "trace_coverage_rate": c6.get("trace_coverage_rate"),
    }
    c6_supported = baseline_false > 0.0 and unsafe_gain is not None
    false_reduction_for_gate = false_reduction if false_reduction is not None else -float("inf")
    unsafe_gain_for_gate = unsafe_gain if unsafe_gain is not None else -float("inf")
    passed_c6 = (
        c6_supported
        and matched_capacity
        and false_reduction_for_gate >= float(cfg["c6_false_auto_clear_relative_reduction"])
        and unsafe_gain_for_gate >= float(cfg["c6_unsafe_capture_gain_pp"])
        and clean_macro_delta is not None
        and clean_macro_delta >= -float(cfg["c6_clean_macro_f1_regression_pp"])
        and float(c6.get("schema_failure_rate", 1.0)) <= float(cfg["max_schema_failure_rate"])
        and float(c6.get("trace_coverage_rate", 0.0)) >= float(cfg["require_trace_coverage"])
    )
    if not c6_supported:
        return {
            "status": "EVIDENCE_UPGRADE_C5_ONLY",
            "reason": "C5 passed. C6 matched-capacity comparison has no residual unsafe confidence-guard events; do not claim RCTR superiority.",
            "gates": {"C5": c5_gates, "C6": c6_gates},
        }
    return {
        "status": "EVIDENCE_UPGRADE_WITH_RCTR" if passed_c6 else "EVIDENCE_UPGRADE_C5_ONLY",
        "reason": (
            "C5 and C6 frozen gates passed."
            if passed_c6
            else "C5 passed; RCTR does not meet the matched-capacity gate."
        ),
        "gates": {"C5": c5_gates, "C6": c6_gates},
    }
