from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from visualact_r.schemas import DetectionCandidate, box_iou


def class_aware_nms(
    candidates: Iterable[DetectionCandidate], iou_threshold: float = 0.5
) -> tuple[list[DetectionCandidate], list[DetectionCandidate]]:
    items = sorted(list(candidates), key=lambda x: x.confidence, reverse=True)
    kept: list[DetectionCandidate] = []
    clustered: list[DetectionCandidate] = []
    cluster_index = 0
    for candidate in items:
        matches = [
            x
            for x in kept
            if x.category == candidate.category
            and box_iou(x.box_xyxy, candidate.box_xyxy) >= iou_threshold
        ]
        if matches:
            anchor = matches[0]
            clustered.append(
                DetectionCandidate(
                    raw_candidate_id=candidate.raw_candidate_id,
                    image_id=candidate.image_id,
                    category=candidate.category,
                    box_xyxy=candidate.box_xyxy,
                    confidence=candidate.confidence,
                    pre_nms=True,
                    nms_cluster_id=anchor.nms_cluster_id,
                    source=candidate.source,
                )
            )
        else:
            cluster_id = f"{candidate.image_id}:cluster:{cluster_index}"
            cluster_index += 1
            anchor = DetectionCandidate(
                raw_candidate_id=candidate.raw_candidate_id,
                image_id=candidate.image_id,
                category=candidate.category,
                box_xyxy=candidate.box_xyxy,
                confidence=candidate.confidence,
                pre_nms=False,
                nms_cluster_id=cluster_id,
                source=candidate.source,
            )
            kept.append(anchor)
            clustered.append(
                DetectionCandidate(
                    raw_candidate_id=candidate.raw_candidate_id,
                    image_id=candidate.image_id,
                    category=candidate.category,
                    box_xyxy=candidate.box_xyxy,
                    confidence=candidate.confidence,
                    pre_nms=True,
                    nms_cluster_id=cluster_id,
                    source=candidate.source,
                )
            )
    return clustered, kept


def candidates_by_cluster(
    candidates: Iterable[DetectionCandidate],
) -> dict[str, list[DetectionCandidate]]:
    grouped: dict[str, list[DetectionCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.nms_cluster_id].append(candidate)
    return grouped
