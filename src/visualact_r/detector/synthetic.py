from __future__ import annotations

from collections import deque

import numpy as np
from PIL import Image

from visualact_r.data.fixtures import CATEGORY_COLORS
from visualact_r.detector.base import class_aware_nms
from visualact_r.schemas import Box, DetectionCandidate, TaskRecord


def _components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    output: list[tuple[int, int, int, int, int]] = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            q = deque([(x, y)])
            visited[y, x] = True
            xs: list[int] = []
            ys: list[int] = []
            while q:
                cx, cy = q.popleft()
                xs.append(cx)
                ys.append(cy)
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if (
                        0 <= nx < width
                        and 0 <= ny < height
                        and mask[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        q.append((nx, ny))
            if len(xs) >= 16:
                output.append((min(xs), min(ys), max(xs) + 1, max(ys) + 1, len(xs)))
    return output


class SyntheticColorDetector:
    """Synthetic fixture detector that reads colored objects from pixels, never GT annotations."""

    source = "synthetic_color_detector_v1"

    def predict(
        self, record: TaskRecord
    ) -> tuple[list[DetectionCandidate], list[DetectionCandidate]]:
        image = np.asarray(Image.open(record.image_path).convert("RGB"))
        raw: list[DetectionCandidate] = []
        idx = 0
        for category, color in CATEGORY_COLORS.items():
            diff = np.abs(image.astype(np.int16) - np.asarray(color, dtype=np.int16))
            mask = np.all(diff <= 5, axis=2)
            for x1, y1, x2, y2, pixels in _components(mask):
                base = Box(float(x1), float(y1), float(x2), float(y2))
                area_factor = min(1.0, base.area / (record.width * record.height * 0.08))
                jitter_scale = 3.0 + (1.0 - area_factor) * 5.0
                for variant, (dx1, dy1, dx2, dy2, confidence) in enumerate(
                    [
                        (0, 0, 0, 0, 0.92),
                        (-jitter_scale, -1, 2, 2, 0.66),
                        (1, -2, jitter_scale, 1, 0.49),
                    ]
                ):
                    box = Box(base.x1 + dx1, base.y1 + dy1, base.x2 + dx2, base.y2 + dy2).clamp(
                        record.width, record.height
                    )
                    raw.append(
                        DetectionCandidate(
                            raw_candidate_id=f"{record.image_id}:raw:{idx}",
                            image_id=record.image_id,
                            category=category,
                            box_xyxy=box,
                            confidence=confidence,
                            pre_nms=True,
                            source=self.source,
                        )
                    )
                    idx += 1
        clustered, nms = class_aware_nms(raw)
        return clustered, nms
