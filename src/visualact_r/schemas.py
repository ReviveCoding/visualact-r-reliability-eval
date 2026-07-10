from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Action = Literal["AUTO_CLEAR", "REVIEW", "ESCALATE"]
ToolName = Literal["INSPECT_REGION", "FINALIZE", "ROUTE_REVIEW"]


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def clamp(self, width: float, height: float) -> "Box":
        x1 = min(max(self.x1, 0.0), width)
        y1 = min(max(self.y1, 0.0), height)
        x2 = min(max(self.x2, x1), width)
        y2 = min(max(self.y2, y1), height)
        return Box(x1, y1, x2, y2)

    def to_dict(self) -> dict[str, float]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Box":
        if "box_xyxy" in payload:
            payload = payload["box_xyxy"]
        return cls(
            float(payload["x1"]), float(payload["y1"]), float(payload["x2"]), float(payload["y2"])
        )


def box_iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - intersection
    return intersection / union if union > 0 else 0.0


def intersection_box(a: Box, b: Box) -> Box | None:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return Box(ix1, iy1, ix2, iy2)


def rectangle_union_area(boxes: list[Box]) -> float:
    """Exact union area of axis-aligned boxes without double-counting overlaps."""
    if not boxes:
        return 0.0
    xs = sorted({x.x1 for x in boxes} | {x.x2 for x in boxes})
    total = 0.0
    for left, right in zip(xs[:-1], xs[1:], strict=True):
        if right <= left:
            continue
        spans = sorted((b.y1, b.y2) for b in boxes if b.x1 < right and b.x2 > left)
        current_start: float | None = None
        current_end: float | None = None
        covered = 0.0
        for start, end in spans:
            if current_start is None:
                current_start, current_end = start, end
            else:
                assert current_end is not None
                if start > current_end:
                    covered += current_end - current_start
                    current_start, current_end = start, end
                else:
                    current_end = max(current_end, end)
        if current_start is not None and current_end is not None:
            covered += current_end - current_start
        total += (right - left) * covered
    return total


@dataclass
class GroundTruthObject:
    object_id: str
    category: str
    box_xyxy: Box
    occluded: bool = False
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["box_xyxy"] = self.box_xyxy.to_dict()
        return out

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GroundTruthObject":
        return cls(
            object_id=str(payload["object_id"]),
            category=str(payload["category"]),
            box_xyxy=Box.from_dict(payload),
            occluded=bool(payload.get("occluded", False)),
            truncated=bool(payload.get("truncated", False)),
        )


@dataclass
class TaskRecord:
    image_id: str
    image_path: str
    width: int
    height: int
    task_id: str
    task_template_id: str
    prompt: str
    weather: str
    scene: str
    timeofday: str
    gt_objects: list[GroundTruthObject] = field(default_factory=list)
    action_label: Action = "AUTO_CLEAR"
    supporting_object_ids: list[str] = field(default_factory=list)
    hard_flags: list[str] = field(default_factory=list)
    phash: str = ""
    phash_cluster_id: str = ""
    split: str = ""
    label_policy_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["gt_objects"] = [x.to_dict() for x in self.gt_objects]
        return out

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskRecord":
        values = dict(payload)
        values["gt_objects"] = [
            GroundTruthObject.from_dict(x) for x in payload.get("gt_objects", [])
        ]
        return cls(**values)

    def inference_contract(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "task_id": self.task_id,
            "task_template_id": self.task_template_id,
            "prompt": self.prompt,
            "split": self.split,
            "label_policy_version": self.label_policy_version,
        }


@dataclass(frozen=True)
class InferenceRecord:
    """Serving-time-safe subset. It intentionally excludes all labels and GT annotations."""

    image_id: str
    image_path: str
    width: int
    height: int
    task_id: str
    task_template_id: str
    prompt: str
    split: str
    label_policy_version: str

    @classmethod
    def from_task_record(cls, record: TaskRecord) -> "InferenceRecord":
        return cls(**record.inference_contract())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "InferenceRecord":
        keys = {
            "image_id",
            "image_path",
            "width",
            "height",
            "task_id",
            "task_template_id",
            "prompt",
            "split",
            "label_policy_version",
        }
        missing = sorted(keys - set(payload))
        if missing:
            raise ValueError(f"Inference record is missing required serving-time fields: {missing}")
        return cls(**{key: payload[key] for key in keys})


@dataclass
class DetectionCandidate:
    raw_candidate_id: str
    image_id: str
    category: str
    box_xyxy: Box
    confidence: float
    pre_nms: bool
    nms_cluster_id: str = ""
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["box_xyxy"] = self.box_xyxy.to_dict()
        return out

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DetectionCandidate":
        values = dict(payload)
        values["box_xyxy"] = Box.from_dict(payload)
        return cls(**values)


@dataclass
class CropRecord:
    crop_id: str
    image_id: str
    query_id: str
    box_xyxy: Box
    primary_candidate_id: str
    raw_candidate_ids: list[str]
    category: str
    base_confidence: float
    edge_expansion: list[float]
    edge_deflation: list[float]
    coverage_proxy: float
    contamination_proxy: float
    selection_rank: int
    variant: str

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["box_xyxy"] = self.box_xyxy.to_dict()
        return out

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CropRecord":
        values = dict(payload)
        values.pop("candidate_id", None)
        values["box_xyxy"] = Box.from_dict(payload)
        return cls(**values)


@dataclass
class AgentStep:
    step: int
    typed_action: ToolName
    region_id: str | None
    decision: Action | None
    confidence: float
    schema_valid: bool
    reason_code: str
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentStep":
        return cls(**payload)


@dataclass
class AgentTrace:
    trace_id: str
    image_id: str
    task_id: str
    candidate_id: str
    visible_crop_ids: list[str]
    steps: list[AgentStep]
    final_decision: Action
    final_confidence: float
    schema_valid: bool
    latency_ms: float
    final_evidence_ids: list[str] = field(default_factory=list)
    review_routed: bool = False
    intervention_step: int | None = None
    risk_score: float | None = None
    rctr_reference_trace_ids: list[str] = field(default_factory=list)
    pre_route_decision: Action | None = None
    pre_route_confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["steps"] = [x.to_dict() for x in self.steps]
        return out

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentTrace":
        values = dict(payload)
        values["steps"] = [AgentStep.from_dict(x) for x in payload.get("steps", [])]
        values.setdefault("final_evidence_ids", [])
        values.setdefault("review_routed", False)
        values.setdefault("intervention_step", None)
        values.setdefault("risk_score", None)
        values.setdefault("rctr_reference_trace_ids", [])
        values.setdefault("pre_route_decision", None)
        values.setdefault("pre_route_confidence", None)
        return cls(**values)
