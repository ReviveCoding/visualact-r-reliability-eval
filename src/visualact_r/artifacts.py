from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from visualact_r.schemas import (
    AgentTrace,
    CropRecord,
    DetectionCandidate,
    InferenceRecord,
    TaskRecord,
)
from visualact_r.utils import read_jsonl, write_jsonl


def load_records(path: str | Path) -> list[TaskRecord]:
    return [TaskRecord.from_dict(x) for x in read_jsonl(path)]


_INFERENCE_FORBIDDEN_FIELDS = frozenset(
    {
        "action_label",
        "supporting_object_ids",
        "gt_objects",
        "weather",
        "scene",
        "timeofday",
        "hard_flags",
        "phash",
        "phash_cluster_id",
    }
)


def load_inference_records(path: str | Path, split: str | None = None) -> list[InferenceRecord]:
    """Load a dedicated serving-time contract and reject label-bearing task-record JSONL.

    Selecting safe fields after reading a full task record is weaker than enforcing a
    separate inference artifact. This loader therefore fails closed whenever the input
    contains ground truth, labels, split-construction metadata, or hard-case metadata.
    """
    output: list[InferenceRecord] = []
    for row_index, row in enumerate(read_jsonl(path), start=1):
        forbidden = sorted(_INFERENCE_FORBIDDEN_FIELDS.intersection(row))
        if forbidden:
            raise ValueError(
                "Inference input contains forbidden non-serving fields at row "
                f"{row_index}: {forbidden}. Pass the generated inference_records.jsonl artifact."
            )
        if split is not None and row.get("split") != split:
            continue
        output.append(InferenceRecord.from_dict(row))
    return output


def save_inference_records(path: str | Path, records: list[InferenceRecord]) -> None:
    write_jsonl(path, [record.to_dict() for record in records])


def load_split_image_ids(path: str | Path, split: str) -> set[str]:
    return {str(row["image_id"]) for row in read_jsonl(path) if row.get("split") == split}


def save_records(path: str | Path, records: list[TaskRecord]) -> None:
    write_jsonl(path, [x.to_dict() for x in records])


def load_candidate_cache(
    raw_path: str | Path, nms_path: str | Path
) -> dict[str, tuple[list[DetectionCandidate], list[DetectionCandidate]]]:
    raw: dict[str, list[DetectionCandidate]] = defaultdict(list)
    nms: dict[str, list[DetectionCandidate]] = defaultdict(list)
    for item in read_jsonl(raw_path):
        candidate = DetectionCandidate.from_dict(item)
        raw[candidate.image_id].append(candidate)
    for item in read_jsonl(nms_path):
        candidate = DetectionCandidate.from_dict(item)
        nms[candidate.image_id].append(candidate)
    return {image_id: (raw[image_id], nms[image_id]) for image_id in sorted(set(raw) | set(nms))}


def save_candidate_cache(
    cache: dict[str, tuple[list[DetectionCandidate], list[DetectionCandidate]]],
    raw_path: str | Path,
    nms_path: str | Path,
) -> None:
    write_jsonl(raw_path, [x.to_dict() for raw, _ in cache.values() for x in raw])
    write_jsonl(nms_path, [x.to_dict() for _, nms in cache.values() for x in nms])


def load_crops(path: str | Path, candidate_id: str | None = None) -> dict[str, list[CropRecord]]:
    out: dict[str, list[CropRecord]] = defaultdict(list)
    for item in read_jsonl(path):
        if candidate_id is not None and item.get("candidate_id") != candidate_id:
            continue
        crop = CropRecord.from_dict(item)
        out[crop.image_id].append(crop)
    return dict(out)


def save_crops(
    path: str | Path, crops_by_image: dict[str, list[CropRecord]], candidate_id: str
) -> None:
    write_jsonl(
        path,
        [
            {**crop.to_dict(), "candidate_id": candidate_id}
            for crops in crops_by_image.values()
            for crop in crops
        ],
    )


def load_traces(path: str | Path, candidate_id: str | None = None) -> list[AgentTrace]:
    out = []
    for item in read_jsonl(path):
        if candidate_id is not None and item.get("candidate_id") != candidate_id:
            continue
        out.append(AgentTrace.from_dict(item))
    return out


def save_traces(path: str | Path, traces: list[AgentTrace]) -> None:
    write_jsonl(path, [x.to_dict() for x in traces])
