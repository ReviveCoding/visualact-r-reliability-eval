from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from visualact_r.detector.base import class_aware_nms
from visualact_r.schemas import Box, DetectionCandidate, TaskRecord


class RawCandidateExtractionUnsupported(RuntimeError):
    pass


class UltralyticsRawCandidateAdapter:
    """Pinned adapter for standard Ultralytics YOLO detection checkpoints.

    The default path decodes the standard Ultralytics raw detection-head tensor, then applies
    project-local class-aware NMS so CEF inputs remain auditable. A bounded post-NMS fallback can
    be enabled for detector smoke tests when a pretrained COCO checkpoint returns no mapped raw
    candidates for a specific image. That fallback is explicitly marked in candidate sources and
    reported by the cache stage; it should not be used for raw-pre-NMS CEF claims without a claim
    boundary note.
    """

    COCO_TO_PROJECT = {0: "person", 1: "bike", 2: "car", 3: "motor", 5: "bus", 7: "truck"}

    def __init__(
        self,
        weights: str,
        imgsz: int = 640,
        confidence: float = 0.10,
        device: str | None = None,
        allow_post_nms_fallback: bool = False,
    ):
        if imgsz <= 0:
            raise ValueError("imgsz must be positive")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        self.weights = weights
        self.imgsz = imgsz
        self.confidence = confidence
        self.device = device
        self.allow_post_nms_fallback = allow_post_nms_fallback
        self._model: Any | None = None
        self.last_used_post_nms_fallback = False

    def _load(self) -> Any:
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "Install visualact-r[vision] to use the Ultralytics adapter"
                ) from exc
            self._model = YOLO(self.weights)
        return self._model

    @staticmethod
    def _xywh_to_xyxy(array: np.ndarray) -> np.ndarray:
        x, y, w, h = [array[:, i] for i in range(4)]
        return np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=1)

    @classmethod
    def _project_category(cls, model: Any, class_id: int) -> str | None:
        """Map a detector's declared class names to the narrow project taxonomy.

        A fine-tuned BDD checkpoint may not use COCO class indices, so names take precedence.
        """
        names = getattr(model, "names", None) or getattr(
            getattr(model, "model", None), "names", None
        )
        raw_name: str | None = None
        if isinstance(names, dict):
            raw_name = names.get(class_id)
        elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            raw_name = str(names[class_id])
        normalized = (raw_name or "").strip().lower().replace(" ", "_")
        aliases = {
            "person": "person",
            "pedestrian": "person",
            "bicycle": "bike",
            "bike": "bike",
            "car": "car",
            "motorcycle": "motor",
            "motor": "motor",
            "bus": "bus",
            "truck": "truck",
        }
        return aliases.get(normalized) or cls.COCO_TO_PROJECT.get(class_id)

    def _raw_pre_nms_candidates(self, model: Any, record: TaskRecord) -> list[DetectionCandidate]:
        try:
            import torch
            from PIL import Image
            from ultralytics.data.augment import LetterBox
            from ultralytics.utils.ops import scale_boxes
        except ImportError as exc:
            raise RuntimeError("Ultralytics runtime dependencies are incomplete") from exc
        image0 = np.asarray(Image.open(record.image_path).convert("RGB"))
        letterbox = LetterBox(new_shape=(self.imgsz, self.imgsz), auto=False, stride=32)
        image = letterbox(image=image0)
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        if self.device:
            tensor = tensor.to(self.device)
        with torch.inference_mode():
            raw_output = model.model(tensor)
        if isinstance(raw_output, (tuple, list)):
            raw_output = raw_output[0]
        if not hasattr(raw_output, "ndim") or raw_output.ndim != 3:
            raise RawCandidateExtractionUnsupported(
                f"Unexpected raw model output type/shape: {type(raw_output)!r}"
            )
        raw_np = raw_output.detach().float().cpu().numpy()
        # Standard Ultralytics detection head returns [B, 4+nc, anchors].
        if raw_np.shape[1] <= raw_np.shape[2]:
            raw_np = np.transpose(raw_np, (0, 2, 1))
        predictions = raw_np[0]
        if predictions.shape[1] < 6:
            raise RawCandidateExtractionUnsupported(
                f"Raw predictions have insufficient channels: {predictions.shape}"
            )
        class_scores = predictions[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        scores = class_scores.max(axis=1)
        keep = np.where(scores >= self.confidence)[0]
        if keep.size == 0:
            return []
        xyxy = self._xywh_to_xyxy(predictions[keep, :4])
        boxes = torch.from_numpy(xyxy)
        boxes = scale_boxes((self.imgsz, self.imgsz), boxes, image0.shape[:2]).numpy()
        candidates: list[DetectionCandidate] = []
        for idx, (box, score, class_id) in enumerate(
            zip(boxes, scores[keep], class_ids[keep], strict=True)
        ):
            category = self._project_category(model, int(class_id))
            if category is None:
                continue
            candidates.append(
                DetectionCandidate(
                    raw_candidate_id=f"{record.image_id}:raw:{idx}",
                    image_id=record.image_id,
                    category=category,
                    box_xyxy=Box(*map(float, box)).clamp(record.width, record.height),
                    confidence=float(score),
                    pre_nms=True,
                    source=f"ultralytics_raw_pre_nms:{Path(self.weights).name}",
                )
            )
        return candidates

    def _post_nms_fallback_candidates(
        self, model: Any, record: TaskRecord
    ) -> tuple[list[DetectionCandidate], list[DetectionCandidate]]:
        results = model.predict(
            source=record.image_path,
            imgsz=self.imgsz,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )
        if not results:
            return [], []
        result = results[0]
        boxes_obj = getattr(result, "boxes", None)
        if boxes_obj is None or getattr(boxes_obj, "xyxy", None) is None:
            return [], []
        xyxy = boxes_obj.xyxy.detach().float().cpu().numpy()
        conf = boxes_obj.conf.detach().float().cpu().numpy()
        cls = boxes_obj.cls.detach().int().cpu().numpy()
        raw: list[DetectionCandidate] = []
        nms: list[DetectionCandidate] = []
        for idx, (box, score, class_id) in enumerate(zip(xyxy, conf, cls, strict=True)):
            category = self._project_category(model, int(class_id))
            if category is None:
                continue
            raw_id = f"{record.image_id}:postnms_fallback:{idx}"
            cluster_id = f"{record.image_id}:postnms_fallback_cluster:{idx}"
            clamped = Box(*map(float, box)).clamp(record.width, record.height)
            raw.append(
                DetectionCandidate(
                    raw_candidate_id=raw_id,
                    image_id=record.image_id,
                    category=category,
                    box_xyxy=clamped,
                    confidence=float(score),
                    pre_nms=True,
                    nms_cluster_id=cluster_id,
                    source=f"ultralytics_post_nms_fallback_as_raw:{Path(self.weights).name}",
                )
            )
            nms.append(
                DetectionCandidate(
                    raw_candidate_id=raw_id,
                    image_id=record.image_id,
                    category=category,
                    box_xyxy=clamped,
                    confidence=float(score),
                    pre_nms=False,
                    nms_cluster_id=cluster_id,
                    source=f"ultralytics_post_nms_fallback:{Path(self.weights).name}",
                )
            )
        return raw, nms

    def predict(
        self, record: TaskRecord
    ) -> tuple[list[DetectionCandidate], list[DetectionCandidate]]:
        self.last_used_post_nms_fallback = False
        model = self._load()
        candidates = self._raw_pre_nms_candidates(model, record)
        if candidates:
            return class_aware_nms(candidates)
        if self.allow_post_nms_fallback:
            raw, nms = self._post_nms_fallback_candidates(model, record)
            if raw:
                self.last_used_post_nms_fallback = True
                return raw, nms
        raise RawCandidateExtractionUnsupported(
            "No raw candidate in project class map; cannot validate CEF input"
        )
