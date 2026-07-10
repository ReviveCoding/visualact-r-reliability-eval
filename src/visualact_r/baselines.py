from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from visualact_r.data.task_policy import normalized_center, point_in_polygon
from visualact_r.utils import atomic_write_bytes
from visualact_r.schemas import Action, DetectionCandidate, TaskRecord

ACTIONS: list[Action] = ["AUTO_CLEAR", "REVIEW", "ESCALATE"]
ACTION_TO_INDEX = {x: i for i, x in enumerate(ACTIONS)}


def detector_features(
    record: TaskRecord, nms: list[DetectionCandidate], config: dict[str, Any]
) -> np.ndarray:
    critical = set(config["data"]["critical_categories"])
    critical_boxes = [x for x in nms if x.category in critical]
    primary = 0
    secondary = 0
    max_area = 0.0
    confidences = []
    for item in critical_boxes:
        x, y = normalized_center(item.box_xyxy, record.width, record.height)
        if point_in_polygon(x, y, config["task_policy"]["primary_zone"]):
            primary += 1
        elif point_in_polygon(x, y, config["task_policy"]["secondary_zone"]):
            secondary += 1
        max_area = max(max_area, item.box_xyxy.area / (record.width * record.height))
        confidences.append(item.confidence)
    return np.asarray(
        [
            len(critical_boxes),
            primary,
            secondary,
            max_area,
            max(confidences) if confidences else 0.0,
            float(np.mean(confidences)) if confidences else 0.0,
        ],
        dtype=np.float32,
    )


def c0_policy(
    record: TaskRecord, nms: list[DetectionCandidate], config: dict[str, Any]
) -> tuple[Action, float]:
    features = detector_features(record, nms, config)
    if (
        features[1] >= 1
        and features[3] >= config["task_policy"]["primary_large_area_fraction"] * 0.65
    ):
        return "ESCALATE", 0.80
    if features[1] >= 1 or features[2] >= 1:
        return "REVIEW", 0.70
    return "AUTO_CLEAR", 0.75


@dataclass
class CalibratedDetectorTriage:
    model: Any | None = None
    fallback_action: Action = "REVIEW"

    def fit(
        self,
        records: list[TaskRecord],
        nms_by_id: dict[str, list[DetectionCandidate]],
        config: dict[str, Any],
    ) -> "CalibratedDetectorTriage":
        x = np.stack(
            [detector_features(record, nms_by_id[record.image_id], config) for record in records]
        )
        y = np.asarray([ACTION_TO_INDEX[record.action_label] for record in records])
        if len(set(y.tolist())) < 2:
            self.fallback_action = ACTIONS[int(y[0])] if len(y) else "REVIEW"
            return self
        base = LogisticRegression(max_iter=500, class_weight="balanced")
        # Small-data-safe: sigmoid calibration with a compact internal CV.
        self.model = CalibratedClassifierCV(base, method="sigmoid", cv=min(3, max(2, len(y) // 4)))
        self.model.fit(x, y)
        return self

    def predict(
        self, record: TaskRecord, nms: list[DetectionCandidate], config: dict[str, Any]
    ) -> tuple[Action, float, np.ndarray]:
        if self.model is None:
            probabilities: Any = np.zeros(3, dtype=float)
            probabilities[ACTION_TO_INDEX[self.fallback_action]] = 1.0
            return self.fallback_action, 1.0, probabilities
        probs = self.model.predict_proba(detector_features(record, nms, config).reshape(1, -1))[0]
        label = ACTIONS[int(np.argmax(probs))]
        return label, float(np.max(probs)), probs

    def save(self, path: str | Path) -> None:
        atomic_write_bytes(path, pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))

    @classmethod
    def load(cls, path: str | Path) -> "CalibratedDetectorTriage":
        with Path(path).open("rb") as handle:
            return pickle.load(handle)
