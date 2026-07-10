from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.metrics import brier_score_loss, f1_score

from visualact_r.schemas import (
    AgentTrace,
    Box,
    CropRecord,
    GroundTruthObject,
    TaskRecord,
    intersection_box,
    rectangle_union_area,
)


def evidence_metrics(crops: list[CropRecord], support: list[GroundTruthObject]) -> dict[str, float]:
    crop_boxes = [x.box_xyxy for x in crops]
    support_boxes = [x.box_xyxy for x in support]
    total_support = rectangle_union_area(support_boxes)
    intersections: list[Box] = []
    for crop in crop_boxes:
        for gt in support_boxes:
            inter = intersection_box(crop, gt)
            if inter is not None:
                intersections.append(inter)
    covered = rectangle_union_area(intersections)
    crop_union = rectangle_union_area(crop_boxes)
    recall = covered / total_support if total_support > 0 else 1.0
    precision = covered / crop_union if crop_union > 0 else (1.0 if total_support == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    redundancy = (sum(x.area for x in crop_boxes) - crop_union) / max(
        1.0, sum(x.area for x in crop_boxes)
    )
    return {
        "critical_evidence_recall": float(recall),
        "evidence_precision": float(precision),
        "evidence_f1": float(f1),
        "evidence_contamination": float(1.0 - precision),
        "crop_redundancy": float(redundancy),
        "mean_crop_count": float(len(crops)),
    }


def expected_calibration_error(
    confidence: np.ndarray, correctness: np.ndarray, bins: int = 10
) -> float:
    if len(confidence) == 0:
        return 0.0
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        mask = (confidence >= lower) & (
            (confidence < upper) if upper < 1.0 else (confidence <= upper)
        )
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(confidence[mask].mean()) - float(correctness[mask].mean())
            )
    return float(ece)


def _support(record: TaskRecord) -> list[GroundTruthObject]:
    return [x for x in record.gt_objects if x.object_id in record.supporting_object_ids]


def _is_hard(record: TaskRecord) -> bool:
    return (
        bool(record.hard_flags)
        or record.timeofday in {"night", "dawn/dusk"}
        or record.weather
        in {
            "rainy",
            "foggy",
        }
    )


def _row_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "macro_f1": None,
            "balanced_accuracy": None,
            "grounded_action_success": None,
            "false_auto_clear_rate": None,
        }
    y_true = [str(row["true_action"]) for row in rows]
    y_pred = [str(row["pred_action"]) for row in rows]
    conf = np.clip(np.asarray([float(row["confidence"]) for row in rows]), 1e-6, 1 - 1e-6)
    correct = np.asarray([int(row["action_correct"]) for row in rows])
    trace_present = np.asarray([int(not row["missing_trace"]) for row in rows])
    unsafe_pre_route = np.asarray([int(row["would_false_auto_clear"]) for row in rows])
    routed_unsafe = np.asarray([int(row["routed_unsafe_capture"]) for row in rows])
    present_recalls = [
        sum(
            1
            for actual, predicted in zip(y_true, y_pred, strict=True)
            if actual == label and predicted == label
        )
        / max(1, sum(1 for actual in y_true if actual == label))
        for label in ["AUTO_CLEAR", "REVIEW", "ESCALATE"]
        if any(actual == label for actual in y_true)
    ]
    return {
        "n": len(rows),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                labels=["AUTO_CLEAR", "REVIEW", "ESCALATE"],
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(np.mean(present_recalls)) if present_recalls else None,
        "grounded_action_success": float(np.mean([row["grounded_action_success"] for row in rows])),
        "false_auto_clear_rate": float(np.mean([row["false_auto_clear"] for row in rows])),
        "safe_case_overflag_rate": float(np.mean([row["safe_case_overflag"] for row in rows])),
        "schema_failure_rate": float(1.0 - np.mean([row["schema_valid"] for row in rows])),
        "trace_coverage_rate": float(np.mean(trace_present)),
        "missing_trace_rate": float(1.0 - np.mean(trace_present)),
        "critical_evidence_recall": float(
            np.mean([row["critical_evidence_recall"] for row in rows])
        ),
        "evidence_precision": float(np.mean([row["evidence_precision"] for row in rows])),
        "evidence_f1": float(np.mean([row["evidence_f1"] for row in rows])),
        "evidence_contamination": float(np.mean([row["evidence_contamination"] for row in rows])),
        "crop_redundancy": float(np.mean([row["crop_redundancy"] for row in rows])),
        "mean_crop_count": float(np.mean([row["mean_crop_count"] for row in rows])),
        "mean_selected_evidence_count": float(
            np.mean([row["selected_evidence_count"] for row in rows])
        ),
        "available_critical_evidence_recall": float(
            np.mean([row["available_critical_evidence_recall"] for row in rows])
        ),
        "available_evidence_precision": float(
            np.mean([row["available_evidence_precision"] for row in rows])
        ),
        "invalid_evidence_id_rate": float(
            np.mean([row["invalid_evidence_id_count"] > 0 for row in rows])
        ),
        "brier_correctness": float(brier_score_loss(correct, conf)),
        "ece_correctness": expected_calibration_error(conf, correct),
        "latency_ms_p50": float(np.percentile([row["latency_ms"] for row in rows], 50)),
        "latency_ms_p95": float(np.percentile([row["latency_ms"] for row in rows], 95)),
        "review_routed_rate": float(np.mean([row["review_routed"] for row in rows])),
        "routed_unsafe_capture_rate": float(routed_unsafe.sum() / unsafe_pre_route.sum())
        if unsafe_pre_route.sum()
        else None,
        "routed_unsafe_precision": float(
            routed_unsafe.sum() / max(1, sum(row["review_routed"] for row in rows))
        ),
        "mean_intervention_step": float(
            np.mean(
                [row["intervention_step"] for row in rows if row["intervention_step"] is not None]
            )
        )
        if any(row["intervention_step"] is not None for row in rows)
        else None,
    }


def evaluate_candidate(
    records: list[TaskRecord],
    traces: list[AgentTrace],
    crops_by_image: dict[str, list[CropRecord]],
    candidate_id: str,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evaluation_cfg = (config or {}).get("evaluation", {})
    required_recall = float(evaluation_cfg.get("grounded_required_recall", 0.80))
    max_contamination = float(evaluation_cfg.get("grounded_max_contamination", 0.70))
    by_image = {x.image_id: x for x in records}
    trace_by_image = {x.image_id: x for x in traces}
    rows: list[dict[str, Any]] = []
    for image_id, record in by_image.items():
        trace = trace_by_image.get(image_id)
        missing_trace = trace is None
        intervention_step: int | None
        if trace is None:
            # Fail closed at evaluation. A missing output is a review fallback, never silently excluded.
            pred_action = "REVIEW"
            confidence = 0.0
            schema_valid = False
            latency_ms = 0.0
            review_routed = True
            intervention_step = 0
            pre_route_decision = "AUTO_CLEAR"
        else:
            pred_action = trace.final_decision
            confidence = trace.final_confidence
            schema_valid = trace.schema_valid
            latency_ms = trace.latency_ms
            review_routed = trace.review_routed
            intervention_step = trace.intervention_step
            pre_route_decision = trace.pre_route_decision or trace.final_decision
        support = _support(record)
        available_crops = crops_by_image.get(image_id, [])
        selected_ids = set(trace.final_evidence_ids) if trace is not None else set()
        visible_ids = set(trace.visible_crop_ids) if trace is not None else set()
        invalid_evidence_ids = sorted(selected_ids - visible_ids)
        selected_crops = [crop for crop in available_crops if crop.crop_id in selected_ids]
        available_evidence = evidence_metrics(available_crops, support)
        evidence = evidence_metrics(selected_crops, support)
        action_ok = pred_action == record.action_label
        support_required = record.action_label != "AUTO_CLEAR"
        evidence_sufficient = not support_required or (
            evidence["critical_evidence_recall"] >= required_recall
            and evidence["evidence_contamination"] <= max_contamination
        )
        would_false_auto_clear = int(
            pre_route_decision == "AUTO_CLEAR" and record.action_label != "AUTO_CLEAR"
        )
        row = {
            "image_id": image_id,
            "candidate_id": candidate_id,
            "split": record.split,
            "final_group": "hard" if _is_hard(record) else "clean",
            "true_action": record.action_label,
            "pred_action": pred_action,
            "pre_route_decision": pre_route_decision,
            "pre_route_confidence": trace.pre_route_confidence
            if trace is not None and trace.pre_route_confidence is not None
            else confidence,
            "confidence": confidence,
            "schema_valid": bool(schema_valid and not invalid_evidence_ids),
            "missing_trace": int(missing_trace),
            "selected_evidence_count": len(selected_crops),
            "visible_evidence_count": len(visible_ids),
            "invalid_evidence_id_count": len(invalid_evidence_ids),
            "action_correct": int(action_ok),
            "evidence_sufficient": int(evidence_sufficient),
            "grounded_action_success": int(action_ok and evidence_sufficient),
            "false_auto_clear": int(
                pred_action == "AUTO_CLEAR" and record.action_label != "AUTO_CLEAR"
            ),
            "would_false_auto_clear": would_false_auto_clear,
            "safe_case_overflag": int(
                pred_action != "AUTO_CLEAR" and record.action_label == "AUTO_CLEAR"
            ),
            "review_routed": int(review_routed),
            "routed_unsafe_capture": int(review_routed and bool(would_false_auto_clear)),
            "intervention_step": intervention_step,
            "timeofday": record.timeofday,
            "weather": record.weather,
            "scene": record.scene,
            "hard_flags": "|".join(record.hard_flags),
            "latency_ms": latency_ms,
            **evidence,
            "available_critical_evidence_recall": available_evidence["critical_evidence_recall"],
            "available_evidence_precision": available_evidence["evidence_precision"],
            "available_evidence_contamination": available_evidence["evidence_contamination"],
        }
        rows.append(row)
    if not rows:
        return {"candidate_id": candidate_id, "n": 0}, rows
    metrics = {"candidate_id": candidate_id, **_row_metrics(rows)}
    for group in ["hard", "clean"]:
        subset = [row for row in rows if row["final_group"] == group]
        group_metrics = _row_metrics(subset)
        for key in ["grounded_action_success", "macro_f1", "false_auto_clear_rate", "n"]:
            metrics[f"{group}_{key}"] = group_metrics.get(key)
    return metrics, rows


def slice_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for column in ["timeofday", "weather", "scene", "final_group"]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row[column])].append(row)
        for value, items in groups.items():
            summary = _row_metrics(items)
            result.append(
                {
                    "slice_dimension": column,
                    "slice_value": value,
                    "n": len(items),
                    "grounded_action_success": summary["grounded_action_success"],
                    "false_auto_clear_rate": summary["false_auto_clear_rate"],
                    "evidence_contamination": summary["evidence_contamination"],
                    "trace_coverage_rate": summary["trace_coverage_rate"],
                }
            )
    flag_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for flag in filter(None, str(row["hard_flags"]).split("|")):
            flag_groups[flag].append(row)
    for flag, items in flag_groups.items():
        summary = _row_metrics(items)
        result.append(
            {
                "slice_dimension": "hard_flag",
                "slice_value": flag,
                "n": len(items),
                "grounded_action_success": summary["grounded_action_success"],
                "false_auto_clear_rate": summary["false_auto_clear_rate"],
                "evidence_contamination": summary["evidence_contamination"],
                "trace_coverage_rate": summary["trace_coverage_rate"],
            }
        )
    return result


def review_capacity_scenarios(
    rows: list[dict[str, Any]],
    risk_by_image: dict[str, float],
    capacities: list[float] = [0.15, 0.20, 0.30],
) -> list[dict[str, Any]]:
    """Evaluate routing from the pre-route AUTO_CLEAR set at a fixed total review capacity."""
    output: list[dict[str, Any]] = []
    for capacity in capacities:
        eligible = [row for row in rows if row["pre_route_decision"] == "AUTO_CLEAR"]
        k = min(len(eligible), int(np.ceil(len(rows) * capacity)))
        ordered = sorted(
            eligible, key=lambda row: (-risk_by_image.get(row["image_id"], 0.0), row["image_id"])
        )
        routed = {row["image_id"] for row in ordered[:k]}
        remaining_eligible = [row for row in eligible if row["image_id"] not in routed]
        unsafe_total = sum(int(row["would_false_auto_clear"]) for row in eligible)
        unsafe_routed = sum(int(row["would_false_auto_clear"]) for row in ordered[:k])
        output.append(
            {
                "capacity": capacity,
                "review_count": len(routed),
                "review_rate": len(routed) / len(rows) if rows else 0.0,
                "eligible_auto_clear_count": len(eligible),
                "false_auto_clear_rate_after_routing": float(
                    np.mean([row["would_false_auto_clear"] for row in remaining_eligible])
                )
                if remaining_eligible
                else 0.0,
                "unsafe_capture_recall": float(unsafe_routed / unsafe_total)
                if unsafe_total
                else None,
                "review_unsafe_precision": float(unsafe_routed / max(1, len(routed))),
                "route_basis": "pre_route_auto_clear_only",
            }
        )
    return output


def paired_bootstrap_delta(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    metric: str,
    final_group: str | None = None,
    iterations: int = 1000,
    seed: int = 20260626,
    min_samples: int = 1,
) -> dict[str, Any]:
    """Paired image-level bootstrap for a candidate delta, preserving matched examples."""
    left = {
        row["image_id"]: row
        for row in left_rows
        if final_group is None or row["final_group"] == final_group
    }
    right = {
        row["image_id"]: row
        for row in right_rows
        if final_group is None or row["final_group"] == final_group
    }
    ids = sorted(set(left) & set(right))
    if not ids or metric not in left[ids[0]] or metric not in right[ids[0]]:
        return {"status": "HOLD", "reason": "no_matched_rows_or_metric", "n": len(ids)}
    if len(ids) < min_samples:
        return {
            "status": "INSUFFICIENT_SAMPLE",
            "reason": "matched_final_sample_count_below_minimum",
            "metric": metric,
            "scope": final_group or "all",
            "left_n": len(left),
            "right_n": len(right),
            "n": len(ids),
            "min_samples": min_samples,
        }
    deltas = np.asarray([float(right[item][metric]) - float(left[item][metric]) for item in ids])
    rng = np.random.default_rng(seed)
    samples: Any = np.empty(iterations, dtype=float)
    for idx in range(iterations):
        picked = rng.integers(0, len(deltas), size=len(deltas))
        samples[idx] = float(np.mean(deltas[picked]))
    return {
        "status": "PASS",
        "metric": metric,
        "scope": final_group or "all",
        "left_n": len(left),
        "right_n": len(right),
        "n": len(ids),
        "delta_mean": float(np.mean(deltas)),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "p_delta_gt_zero": float(np.mean(samples > 0.0)),
        "iterations": iterations,
        "seed": seed,
    }
