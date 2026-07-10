from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from visualact_r.utils import atomic_write_bytes
from visualact_r.schemas import AgentStep, AgentTrace, CropRecord, TaskRecord


PREFIX_FEATURE_NAMES = [
    "prefix_step_count",
    "last_step_confidence",
    "prefix_schema_valid",
    "inspect_call_count",
    "selected_crop_count",
    "mean_selected_coverage_proxy",
    "max_selected_coverage_proxy",
    "mean_selected_contamination_proxy",
    "repeated_region_fraction",
    "visible_crop_count",
]


def _selected_crop_ids(trace: AgentTrace, prefix_len: int) -> list[str]:
    return [
        step.region_id
        for step in trace.steps[:prefix_len]
        if step.typed_action == "INSPECT_REGION" and step.region_id is not None
    ]


def prefix_features(
    trace: AgentTrace, crops: list[CropRecord], prefix_len: int | None = None
) -> np.ndarray:
    """Build an intervention-time feature vector from a trace prefix only.

    It never reads `final_decision`, `final_confidence`, a post-prefix step, or an evaluation label. This is the
    crucial PHT-derived state contract: the critic may learn from completed train trajectories but can only
    observe what the agent had seen at the actual intervention point.
    """
    if prefix_len is None:
        # By default use the last pre-final action, not the terminal action itself.
        prefix_len = max(0, len(trace.steps) - 1)
    if prefix_len < 0 or prefix_len > len(trace.steps):
        raise ValueError("prefix_len is outside trace bounds")
    prefix = trace.steps[:prefix_len]
    last = prefix[-1] if prefix else None
    selected_ids = _selected_crop_ids(trace, prefix_len)
    selected = [crop for crop in crops if crop.crop_id in set(selected_ids)]
    region_ids = [x for x in selected_ids if x]
    repeated_fraction = 0.0
    if region_ids:
        repeated_fraction = 1.0 - len(set(region_ids)) / len(region_ids)
    return np.asarray(
        [
            len(prefix),
            float(last.confidence if last else 0.0),
            float(all(step.schema_valid for step in prefix)),
            float(sum(step.typed_action == "INSPECT_REGION" for step in prefix)),
            float(len(selected)),
            float(np.mean([x.coverage_proxy for x in selected])) if selected else 0.0,
            float(np.max([x.coverage_proxy for x in selected])) if selected else 0.0,
            float(np.mean([x.contamination_proxy for x in selected])) if selected else 1.0,
            repeated_fraction,
            float(len(trace.visible_crop_ids)),
        ],
        dtype=np.float32,
    )


def _terminal_unsafe(trace: AgentTrace, expected_action: str) -> int:
    return int(
        not trace.schema_valid
        or trace.final_decision != expected_action
        or (trace.final_decision == "AUTO_CLEAR" and expected_action != "AUTO_CLEAR")
    )


def _predecision_prefix_lengths(trace: AgentTrace) -> list[int]:
    """Return only prefixes before the agent takes its terminal action or review-route action."""
    lengths: list[int] = []
    for idx, step in enumerate(trace.steps, start=1):
        if step.typed_action in {"FINALIZE", "ROUTE_REVIEW"}:
            break
        lengths.append(idx)
    return lengths


def _diverse_reference_indices(
    features: np.ndarray, action_labels: list[str], max_references: int
) -> list[int]:
    """Action-stratified farthest-point selection for a compact, non-redundant reference library."""
    if len(features) <= max_references:
        return list(range(len(features)))
    selected: list[int] = []
    unique_actions = sorted(set(action_labels))
    base_quota = max(1, max_references // max(1, len(unique_actions)))
    for action in unique_actions:
        candidates = [idx for idx, value in enumerate(action_labels) if value == action]
        if not candidates:
            continue
        chosen = [candidates[0]]
        while len(chosen) < min(base_quota, len(candidates)):
            remaining = [idx for idx in candidates if idx not in chosen]
            distances = [
                min(float(np.linalg.norm(features[idx] - features[old])) for old in chosen)
                for idx in remaining
            ]
            chosen.append(remaining[int(np.argmax(distances))])
        selected.extend(chosen)
    if len(selected) >= max_references:
        return selected[:max_references]
    remaining = [idx for idx in range(len(features)) if idx not in selected]
    while remaining and len(selected) < max_references:
        if not selected:
            selected.append(remaining.pop(0))
            continue
        score = [
            min(float(np.linalg.norm(features[idx] - features[old])) for old in selected)
            for idx in remaining
        ]
        pick_pos = int(np.argmax(score))
        selected.append(remaining.pop(pick_pos))
    return selected


def _mean_nearest_distance(
    references: np.ndarray, row: np.ndarray, n_reference: int, exclude: int | None = None
) -> float:
    if len(references) == 0:
        return 0.0
    distances = np.linalg.norm(references - row, axis=1)
    if exclude is not None and 0 <= exclude < len(distances):
        distances = np.delete(distances, exclude)
    if len(distances) == 0:
        return 0.0
    return float(np.mean(np.sort(distances)[: min(n_reference, len(distances))]))


@dataclass
class RCTRModel:
    scaler: Any | None
    classifier: Any | None
    references: np.ndarray
    reference_prefix_ids: list[str]
    feature_names: list[str]
    fallback_risk: float = 1.0

    @property
    def reference_trace_ids(self) -> list[str]:
        return sorted({prefix_id.rsplit("@", 1)[0] for prefix_id in self.reference_prefix_ids})

    def save(self, path: str | Path) -> None:
        atomic_write_bytes(path, pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))

    @classmethod
    def load(cls, path: str | Path) -> "RCTRModel":
        with Path(path).open("rb") as handle:
            return pickle.load(handle)

    def risk(
        self,
        trace: AgentTrace,
        crops: list[CropRecord],
        n_reference: int = 8,
        prefix_len: int | None = None,
    ) -> tuple[float, list[str]]:
        feature = prefix_features(trace, crops, prefix_len=prefix_len).reshape(1, -1)
        if self.scaler is None or self.classifier is None or len(self.references) == 0:
            return self.fallback_risk, []
        transformed = self.scaler.transform(feature)
        distance = np.linalg.norm(self.references - transformed, axis=1)
        indices = np.argsort(distance)[: min(n_reference, len(distance))]
        reference_distance = float(np.mean(distance[indices])) if len(indices) else 0.0
        probability = float(
            self.classifier.predict_proba(np.hstack([transformed, [[reference_distance]]]))[0, 1]
        )
        return probability, [self.reference_prefix_ids[int(i)] for i in indices]


def fit_rctr(
    train_records: list[TaskRecord],
    train_traces: list[AgentTrace],
    crops_by_image: dict[str, list[CropRecord]],
    n_reference: int = 32,
    neighbor_count: int = 8,
) -> RCTRModel:
    labels = {x.image_id: x.action_label for x in train_records}
    examples: list[tuple[str, np.ndarray, int, str]] = []
    reference_candidates: list[tuple[str, np.ndarray, str]] = []
    for trace in train_traces:
        expected = labels.get(trace.image_id)
        if expected is None:
            continue
        unsafe = _terminal_unsafe(trace, expected)
        for prefix_len in _predecision_prefix_lengths(trace):
            feature = prefix_features(trace, crops_by_image.get(trace.image_id, []), prefix_len)
            prefix_id = f"{trace.trace_id}@{prefix_len}"
            examples.append((prefix_id, feature, unsafe, expected))
            if unsafe == 0:
                reference_candidates.append((prefix_id, feature, expected))
    if not reference_candidates:
        return RCTRModel(
            None, None, np.empty((0, len(PREFIX_FEATURE_NAMES))), [], PREFIX_FEATURE_NAMES
        )
    raw_ref = np.stack([x[1] for x in reference_candidates])
    scaler = StandardScaler().fit(raw_ref)
    transformed_ref_all = scaler.transform(raw_ref)
    selected_idx = _diverse_reference_indices(
        transformed_ref_all,
        [x[2] for x in reference_candidates],
        max_references=max(1, n_reference),
    )
    references = transformed_ref_all[selected_idx]
    reference_prefix_ids = [reference_candidates[idx][0] for idx in selected_idx]
    if not examples:
        return RCTRModel(
            scaler,
            None,
            references,
            reference_prefix_ids,
            PREFIX_FEATURE_NAMES,
            fallback_risk=0.0,
        )
    raw_x = np.stack([x[1] for x in examples])
    y = np.asarray([x[2] for x in examples])
    transformed = scaler.transform(raw_x)
    ref_index = {prefix_id: idx for idx, prefix_id in enumerate(reference_prefix_ids)}
    distances = np.asarray(
        [
            _mean_nearest_distance(references, row, neighbor_count, ref_index.get(prefix_id))
            for prefix_id, row in zip([x[0] for x in examples], transformed, strict=True)
        ]
    ).reshape(-1, 1)
    if len(set(y.tolist())) < 2:
        return RCTRModel(
            scaler,
            None,
            references,
            reference_prefix_ids,
            PREFIX_FEATURE_NAMES,
            fallback_risk=float(y[0]) if len(y) else 1.0,
        )
    classifier = LogisticRegression(
        solver="liblinear",
        max_iter=200,
        class_weight="balanced",
        random_state=0,
    ).fit(np.hstack([transformed, distances]), y)
    return RCTRModel(scaler, classifier, references, reference_prefix_ids, PREFIX_FEATURE_NAMES)


def select_review_routes(
    traces: list[AgentTrace], risk_by_image: dict[str, float], review_capacity: float
) -> set[str]:
    """Select a batch-constrained set of AUTO_CLEAR proposals without using outcomes or labels."""
    if not 0.0 <= review_capacity <= 1.0:
        raise ValueError("review_capacity must be within [0, 1]")
    eligible = [trace for trace in traces if trace.final_decision == "AUTO_CLEAR"]
    k = min(len(eligible), int(np.ceil(len(traces) * review_capacity)))
    ordered = sorted(
        eligible, key=lambda trace: (-risk_by_image.get(trace.image_id, 0.0), trace.image_id)
    )
    return {trace.image_id for trace in ordered[:k]}


def apply_review_routes(
    traces: list[AgentTrace],
    routed_image_ids: set[str],
    risk_by_image: dict[str, float],
    refs: dict[str, list[str]],
    candidate_id: str = "C6",
    reason_code: str = "rctr_prefix_risk_capacity_route",
) -> list[AgentTrace]:
    """Insert a review route before the terminal AUTO_CLEAR is acted on.

    The retained terminal step is stored for auditability, but the C6 final action is REVIEW and the intervention
    metadata records that the routing decision was made at the pre-final prefix.
    """
    out: list[AgentTrace] = []
    for trace in traces:
        risk = float(risk_by_image.get(trace.image_id, 0.0))
        if trace.image_id not in routed_image_ids:
            out.append(
                AgentTrace(
                    trace_id=f"{trace.image_id}:{candidate_id}",
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
                    review_routed=False,
                    intervention_step=None,
                    risk_score=risk,
                    rctr_reference_trace_ids=refs.get(trace.image_id, []),
                    pre_route_decision=trace.final_decision,
                    pre_route_confidence=trace.final_confidence,
                )
            )
            continue
        routed_step = AgentStep(
            step=max(1, len(trace.steps)),
            typed_action="ROUTE_REVIEW",
            region_id=None,
            decision="REVIEW",
            confidence=float(max(0.0, min(1.0, 1.0 - risk))),
            schema_valid=True,
            reason_code=reason_code,
            latency_ms=0.0,
        )
        # The route is the terminal product action. Keep only pre-decision evidence followed by the route.
        prefix = [
            step for step in trace.steps if step.typed_action not in {"FINALIZE", "ROUTE_REVIEW"}
        ]
        out.append(
            AgentTrace(
                trace_id=f"{trace.image_id}:{candidate_id}",
                image_id=trace.image_id,
                task_id=trace.task_id,
                candidate_id=candidate_id,
                visible_crop_ids=trace.visible_crop_ids,
                steps=prefix + [routed_step],
                final_decision="REVIEW",
                final_confidence=float(max(0.0, min(1.0, 1.0 - risk))),
                schema_valid=trace.schema_valid,
                latency_ms=trace.latency_ms,
                final_evidence_ids=trace.final_evidence_ids,
                review_routed=True,
                intervention_step=len(prefix) + 1,
                risk_score=risk,
                rctr_reference_trace_ids=refs.get(trace.image_id, []),
                pre_route_decision=trace.final_decision,
                pre_route_confidence=trace.final_confidence,
            )
        )
    return out
