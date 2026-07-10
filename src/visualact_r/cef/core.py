from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from visualact_r.data.task_policy import normalized_center, point_in_polygon
from visualact_r.detector.base import candidates_by_cluster
from visualact_r.utils import atomic_write_bytes
from visualact_r.schemas import (
    Box,
    CropRecord,
    DetectionCandidate,
    GroundTruthObject,
    TaskRecord,
    box_iou,
    intersection_box,
    rectangle_union_area,
)


def _weighted_quantile(values: list[float], weights: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    order = np.argsort(values)
    arr = np.asarray(values)[order]
    w = np.asarray(weights)[order]
    cumulative = np.cumsum(w) / max(1e-8, np.sum(w))
    index = min(int(np.searchsorted(cumulative, quantile)), len(arr) - 1)
    return float(arr[index])


def _anchor_features(
    anchor: DetectionCandidate, members: list[DetectionCandidate], width: int, height: int
) -> np.ndarray:
    box = anchor.box_xyxy
    xs1 = [m.box_xyxy.x1 for m in members]
    ys1 = [m.box_xyxy.y1 for m in members]
    xs2 = [m.box_xyxy.x2 for m in members]
    ys2 = [m.box_xyxy.y2 for m in members]
    conf = [max(1e-4, m.confidence) for m in members]
    return np.asarray(
        [
            box.x1 / width,
            box.y1 / height,
            box.x2 / width,
            box.y2 / height,
            box.area / (width * height),
            anchor.confidence,
            len(members),
            max(0.0, box.x1 - min(xs1)) / width,
            max(0.0, max(xs2) - box.x2) / width,
            max(0.0, box.y1 - min(ys1)) / height,
            max(0.0, max(ys2) - box.y2) / height,
            float(np.std(xs1)) / width,
            float(np.std(ys1)) / height,
            float(np.std(xs2)) / width,
            float(np.std(ys2)) / height,
            float(np.mean(conf)),
            float(np.max(conf)),
        ],
        dtype=np.float32,
    )


def inflate_box(
    anchor: DetectionCandidate,
    members: list[DetectionCandidate],
    width: int,
    height: int,
    strength: float,
) -> tuple[Box, list[float]]:
    base = anchor.box_xyxy
    weights = [max(1e-4, x.confidence) for x in members]
    left = _weighted_quantile([x.box_xyxy.x1 for x in members], weights, 0.15)
    right = _weighted_quantile([x.box_xyxy.x2 for x in members], weights, 0.85)
    top = _weighted_quantile([x.box_xyxy.y1 for x in members], weights, 0.15)
    bottom = _weighted_quantile([x.box_xyxy.y2 for x in members], weights, 0.85)
    expanded = Box(
        base.x1 - strength * max(0.0, base.x1 - left),
        base.y1 - strength * max(0.0, base.y1 - top),
        base.x2 + strength * max(0.0, right - base.x2),
        base.y2 + strength * max(0.0, bottom - base.y2),
    ).clamp(width, height)
    expansion = [
        base.x1 - expanded.x1,
        base.y1 - expanded.y1,
        expanded.x2 - base.x2,
        expanded.y2 - base.y2,
    ]
    return expanded, expansion


def _best_gt(
    anchor: DetectionCandidate, objects: list[GroundTruthObject]
) -> GroundTruthObject | None:
    same = [x for x in objects if x.category == anchor.category]
    if not same:
        return None
    best = max(same, key=lambda x: box_iou(anchor.box_xyxy, x.box_xyxy))
    return best if box_iou(anchor.box_xyxy, best.box_xyxy) > 0.01 else None


def _support_quality(box: Box, support: list[GroundTruthObject]) -> tuple[float, float]:
    """Return coarse support coverage and contamination for train-only crop ranking labels.

    This approximation intentionally has no role at inference. Exact union-aware evidence metrics are computed
    later by the evaluation package on held-out data.
    """
    if not support:
        return 1.0, 0.0
    support_area = max(1.0, rectangle_union_area([item.box_xyxy for item in support]))
    overlaps = [
        inter for item in support if (inter := intersection_box(box, item.box_xyxy)) is not None
    ]
    overlap = min(rectangle_union_area(overlaps), support_area)
    coverage = overlap / support_area
    precision = overlap / max(1.0, box.area)
    return float(coverage), float(1.0 - min(1.0, precision))


def _rank_label(box: Box, support: list[GroundTruthObject]) -> int:
    coverage, contamination = _support_quality(box, support)
    # This makes the ranker prefer decision-supporting evidence rather than any tiny IoU overlap.
    return int(coverage >= 0.25 and contamination <= 0.88)


@dataclass
class DirectionalBiasCorrector:
    models: list[Any]
    max_correction_fraction: float
    shrinkage: float

    @classmethod
    def fit(
        cls,
        rows: list[tuple[np.ndarray, np.ndarray]],
        max_correction_fraction: float,
        shrinkage: float,
    ) -> "DirectionalBiasCorrector":
        if len(rows) < 4:
            return cls(
                models=[], max_correction_fraction=max_correction_fraction, shrinkage=shrinkage
            )
        x = np.stack([r[0] for r in rows])
        y = np.stack([r[1] for r in rows])
        models = [Ridge(alpha=1.0).fit(x, y[:, idx]) for idx in range(4)]
        return cls(
            models=models, max_correction_fraction=max_correction_fraction, shrinkage=shrinkage
        )

    def correct(
        self, inflated: Box, features: np.ndarray, width: int, height: int
    ) -> tuple[Box, list[float]]:
        if not self.models:
            return inflated, [0.0, 0.0, 0.0, 0.0]
        deltas = np.asarray(
            [model.predict(features.reshape(1, -1))[0] for model in self.models], dtype=np.float32
        )
        scales: Any = np.asarray([width, height, width, height], dtype=np.float32)
        max_delta = self.max_correction_fraction * scales
        deltas = np.clip(deltas * self.shrinkage, -max_delta, max_delta)
        # Target residual is GT edge minus inflated edge, so signs preserve the edge geometry.
        corrected = Box(
            inflated.x1 + deltas[0],
            inflated.y1 + deltas[1],
            inflated.x2 + deltas[2],
            inflated.y2 + deltas[3],
        ).clamp(width, height)
        return corrected, deltas.tolist()


@dataclass
class CropRanker:
    model: Any | None = None
    constant: float | None = None

    def fit(self, x: list[np.ndarray], y: list[int]) -> "CropRanker":
        if not x or len(set(y)) < 2:
            self.constant = float(y[0]) if y else 0.0
            return self
        self.model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500, class_weight="balanced")),
            ]
        )
        self.model.fit(np.stack(x), np.asarray(y))
        return self

    def score(self, feature: np.ndarray) -> float:
        if self.model is not None:
            return float(self.model.predict_proba(feature.reshape(1, -1))[0, 1])
        return float(self.constant or 0.0)


@dataclass
class CEFModel:
    corrector: DirectionalBiasCorrector
    ranker: CropRanker
    raw_ranker: CropRanker
    config: dict[str, Any]
    development_gate: dict[str, Any] = field(default_factory=dict)
    cef_enabled: bool = True

    def save(self, path: str | Path) -> None:
        atomic_write_bytes(path, pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))

    @classmethod
    def load(cls, path: str | Path) -> "CEFModel":
        with Path(path).open("rb") as handle:
            return pickle.load(handle)


def crop_feature(
    anchor: DetectionCandidate,
    box: Box,
    expansion: list[float],
    deflation: list[float],
    record: TaskRecord,
    config: dict[str, Any],
) -> np.ndarray:
    x, y = normalized_center(box, record.width, record.height)
    primary = point_in_polygon(x, y, config["task_policy"]["primary_zone"])
    secondary = point_in_polygon(x, y, config["task_policy"]["secondary_zone"])
    category_code = float(
        sorted(config["data"]["critical_categories"]).index(anchor.category)
        if anchor.category in config["data"]["critical_categories"]
        else -1
    )
    return np.asarray(
        [
            anchor.confidence,
            box.area / (record.width * record.height),
            x,
            y,
            float(primary),
            float(secondary),
            category_code,
            *[value / max(record.width, record.height) for value in expansion],
            *[value / max(record.width, record.height) for value in deflation],
        ],
        dtype=np.float32,
    )


def fit_cef(
    records: list[TaskRecord],
    cache: dict[str, tuple[list[DetectionCandidate], list[DetectionCandidate]]],
    config: dict[str, Any],
) -> CEFModel:
    cef_cfg = config["cef"]
    correction_rows: list[tuple[np.ndarray, np.ndarray]] = []
    for record in records:
        raw, nms = cache[record.image_id]
        grouped = candidates_by_cluster(raw)
        for anchor in nms:
            members = grouped[anchor.nms_cluster_id]
            features = _anchor_features(anchor, members, record.width, record.height)
            inflated, _ = inflate_box(
                anchor, members, record.width, record.height, float(cef_cfg["expansion_strength"])
            )
            target = _best_gt(anchor, record.gt_objects)
            if target is not None:
                residual: Any = np.asarray(
                    [
                        target.box_xyxy.x1 - inflated.x1,
                        target.box_xyxy.y1 - inflated.y1,
                        target.box_xyxy.x2 - inflated.x2,
                        target.box_xyxy.y2 - inflated.y2,
                    ],
                    dtype=np.float32,
                )
                correction_rows.append((features, residual))
    corrector = DirectionalBiasCorrector.fit(
        correction_rows,
        max_correction_fraction=float(cef_cfg["max_correction_fraction"]),
        shrinkage=float(cef_cfg["correction_shrinkage"]),
    )
    cef_x: list[np.ndarray] = []
    cef_y: list[int] = []
    raw_x: list[np.ndarray] = []
    raw_y: list[int] = []
    for record in records:
        raw, nms = cache[record.image_id]
        grouped = candidates_by_cluster(raw)
        support = [x for x in record.gt_objects if x.object_id in record.supporting_object_ids]
        for anchor in nms:
            members = grouped[anchor.nms_cluster_id]
            base_features = _anchor_features(anchor, members, record.width, record.height)
            inflated, expansion = inflate_box(
                anchor, members, record.width, record.height, float(cef_cfg["expansion_strength"])
            )
            corrected, deflation = corrector.correct(
                inflated, base_features, record.width, record.height
            )
            cef_x.append(crop_feature(anchor, corrected, expansion, deflation, record, config))
            cef_y.append(_rank_label(corrected, support))
            raw_x.append(
                crop_feature(anchor, anchor.box_xyxy, [0.0] * 4, [0.0] * 4, record, config)
            )
            raw_y.append(_rank_label(anchor.box_xyxy, support))
    return CEFModel(
        corrector=corrector,
        ranker=CropRanker().fit(cef_x, cef_y),
        raw_ranker=CropRanker().fit(raw_x, raw_y),
        config=config,
    )


def _crop_from_anchor(
    record: TaskRecord,
    anchor: DetectionCandidate,
    members: list[DetectionCandidate],
    variant: str,
    config: dict[str, Any],
    model: CEFModel | None,
) -> tuple[float, CropRecord]:
    inflated, expansion = inflate_box(
        anchor, members, record.width, record.height, float(config["cef"]["expansion_strength"])
    )
    corrected, deflation = inflated, [0.0, 0.0, 0.0, 0.0]
    effective_variant = variant
    if variant == "cef" and model is not None and model.cef_enabled:
        corrected, deflation = model.corrector.correct(
            inflated,
            _anchor_features(anchor, members, record.width, record.height),
            record.width,
            record.height,
        )
    elif variant == "cef" and model is not None and not model.cef_enabled:
        corrected = anchor.box_xyxy
        expansion = [0.0, 0.0, 0.0, 0.0]
        effective_variant = "cef_dev_fallback_nms"
    elif variant in {"nms", "raw_ranker"}:
        corrected = anchor.box_xyxy
        expansion = [0.0, 0.0, 0.0, 0.0]
    elif variant == "inflation":
        corrected = inflated
    else:
        if variant not in {"cef", "nms", "raw_ranker", "inflation"}:
            raise ValueError(f"Unknown crop variant: {variant}")
    feature = crop_feature(anchor, corrected, expansion, deflation, record, config)
    if variant == "cef" and model is not None and model.cef_enabled:
        score = model.ranker.score(feature)
    elif variant == "raw_ranker" and model is not None:
        score = model.raw_ranker.score(feature)
    else:
        x, y = normalized_center(corrected, record.width, record.height)
        zone_bonus = 0.15 * float(point_in_polygon(x, y, config["task_policy"]["primary_zone"]))
        score = anchor.confidence + zone_bonus
    crop = CropRecord(
        crop_id=f"{record.image_id}:{effective_variant}:{anchor.nms_cluster_id}",
        image_id=record.image_id,
        query_id=record.task_id,
        box_xyxy=corrected,
        primary_candidate_id=anchor.raw_candidate_id,
        raw_candidate_ids=[x.raw_candidate_id for x in members],
        category=anchor.category,
        base_confidence=anchor.confidence,
        edge_expansion=expansion,
        edge_deflation=deflation,
        coverage_proxy=float(score),
        contamination_proxy=max(0.0, corrected.area - anchor.box_xyxy.area)
        / max(1.0, corrected.area),
        selection_rank=0,
        variant=effective_variant,
    )
    return score, crop


def _diverse_top_k(
    scored: list[tuple[float, CropRecord]], top_k: int, penalty: float
) -> list[CropRecord]:
    """Greedy MMR-style selection so a Top-K crop set does not waste its evidence budget."""
    remaining = list(scored)
    chosen: list[CropRecord] = []
    while remaining and len(chosen) < top_k:

        def marginal(item: tuple[float, CropRecord]) -> tuple[float, float, str]:
            score, crop = item
            overlap = max((box_iou(crop.box_xyxy, old.box_xyxy) for old in chosen), default=0.0)
            return (score - penalty * overlap, score, crop.crop_id)

        best = max(remaining, key=marginal)
        chosen.append(best[1])
        remaining.remove(best)
    for rank, crop in enumerate(chosen, start=1):
        crop.selection_rank = rank
    return chosen


def make_crops(
    record: TaskRecord,
    raw: list[DetectionCandidate],
    nms: list[DetectionCandidate],
    variant: str,
    config: dict[str, Any],
    model: CEFModel | None = None,
) -> list[CropRecord]:
    grouped = candidates_by_cluster(raw)
    scored = [
        _crop_from_anchor(record, anchor, grouped[anchor.nms_cluster_id], variant, config, model)
        for anchor in nms
    ]
    scored.sort(key=lambda item: (-item[0], item[1].crop_id))
    if variant in {"cef", "raw_ranker"}:
        return _diverse_top_k(
            scored,
            top_k=int(config["cef"]["top_k"]),
            penalty=float(config["cef"].get("diversity_iou_penalty", 0.30)),
        )
    out = [crop for _, crop in scored[: int(config["cef"]["top_k"])]]
    for rank, crop in enumerate(out, start=1):
        crop.selection_rank = rank
    return out


def _aggregate_evidence(
    crops: list[CropRecord], support: list[GroundTruthObject]
) -> dict[str, float]:
    """Measure CEF development coverage with exact rectangle unions.

    Candidate crops and support boxes can overlap. Summing pairwise areas would overstate
    coverage and understate contamination, potentially enabling CEF on a biased dev gate.
    The release evaluation uses union-area semantics, so the development gate must match it.
    """
    if not support:
        return {"recall": 1.0, "contamination": 0.0}
    support_boxes = [item.box_xyxy for item in support]
    crop_boxes = [crop.box_xyxy for crop in crops]
    support_area = max(1.0, rectangle_union_area(support_boxes))
    crop_area = rectangle_union_area(crop_boxes)
    intersections = [
        overlap
        for crop_box in crop_boxes
        for support_box in support_boxes
        if (overlap := intersection_box(crop_box, support_box)) is not None
    ]
    overlap_area = rectangle_union_area(intersections)
    return {
        "recall": float(min(1.0, overlap_area / support_area)),
        "contamination": float(1.0 - min(1.0, overlap_area / max(1.0, crop_area))),
    }


def evaluate_development_gate(
    model: CEFModel,
    dev_records: list[TaskRecord],
    cache: dict[str, tuple[list[DetectionCandidate], list[DetectionCandidate]]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Decide, using development only, whether CEF may replace NMS crops downstream.

    The gate is intentionally conservative. A failed gate makes the C5 serving path fall back to conventional
    NMS crops while preserving the failed CEF artifact for transparent analysis.
    """
    if not dev_records:
        model.cef_enabled = False
        return {"status": "HOLD", "reason": "no_dev_records", "cef_enabled": False}
    nms_rows = []
    cef_rows = []
    for record in dev_records:
        raw, nms = cache[record.image_id]
        support = [x for x in record.gt_objects if x.object_id in record.supporting_object_ids]
        nms_rows.append(_aggregate_evidence(make_crops(record, raw, nms, "nms", config), support))
        # Force the proposed mechanism for evaluation before deciding whether to enable it.
        enabled = model.cef_enabled
        model.cef_enabled = True
        cef_rows.append(
            _aggregate_evidence(make_crops(record, raw, nms, "cef", config, model), support)
        )
        model.cef_enabled = enabled
    nms_recall = float(np.mean([x["recall"] for x in nms_rows]))
    cef_recall = float(np.mean([x["recall"] for x in cef_rows]))
    nms_contam = float(np.mean([x["contamination"] for x in nms_rows]))
    cef_contam = float(np.mean([x["contamination"] for x in cef_rows]))
    recall_gain_pp = 100.0 * (cef_recall - nms_recall)
    contamination_delta_pp = 100.0 * (cef_contam - nms_contam)
    gate_cfg = config["cef"]
    enabled = recall_gain_pp >= float(
        gate_cfg.get("development_min_recall_gain_pp", 0.0)
    ) and contamination_delta_pp <= float(
        gate_cfg.get("development_contamination_tolerance_pp", 2.0)
    )
    model.cef_enabled = enabled
    return {
        "status": "PASS" if enabled else "HOLD",
        "cef_enabled": enabled,
        "n_dev": len(dev_records),
        "nms_critical_evidence_recall": nms_recall,
        "cef_critical_evidence_recall": cef_recall,
        "nms_evidence_contamination": nms_contam,
        "cef_evidence_contamination": cef_contam,
        "recall_gain_pp": recall_gain_pp,
        "contamination_delta_pp": contamination_delta_pp,
        "reason": "development_only_coverage_contamination_gate",
    }
