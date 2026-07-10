from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import os
import tempfile
from typing import Any

from PIL import Image

from visualact_r.artifacts import load_crops, load_records
from visualact_r.schemas import CropRecord, InferenceRecord, TaskRecord, box_iou
from visualact_r.utils import ensure_dir, read_jsonl, stable_hash, utc_now, write_json, write_jsonl

SYSTEM_PROMPT = (
    "You are an offline urban-scene assessment assistant. The allowed output is compact JSON for "
    "AUTO_CLEAR, REVIEW, or ESCALATE triage. This is not vehicle control, collision prediction, or a "
    "safety certification task. Use only supplied images and crop identifiers."
)


@dataclass(frozen=True)
class SFTExample:
    example_id: str
    image_id: str
    split: str
    candidate_id: str
    image_paths: list[str]
    prompt: str
    target: str
    label_policy_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "image_id": self.image_id,
            "split": self.split,
            "candidate_id": self.candidate_id,
            "image_paths": self.image_paths,
            "prompt": self.prompt,
            "target": self.target,
            "system": SYSTEM_PROMPT,
            "label_policy_version": self.label_policy_version,
        }


_WINDOWS_FORBIDDEN_FILENAME_CHARS = frozenset('<>:"/\\|?*')
_WINDOWS_RESERVED_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)
_SAFE_CROP_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})


def _safe_name(value: str, *, fallback: str = "crop", max_length: int = 96) -> str:
    """Return a deterministic filename component that is valid on Windows and POSIX filesystems.

    Crop identifiers are provenance keys and may contain separators such as ``:``. They must not be written
    directly to a filename because Windows reserves ``<>:\"/\\|?*`` and rejects control characters and trailing
    spaces or periods. A stable suffix avoids collisions after sanitization and bounds filename length.
    """
    original = str(value)
    cleaned = (
        "".join(
            "_"
            if character in _WINDOWS_FORBIDDEN_FILENAME_CHARS or ord(character) < 32
            else character
            for character in original
        )
        .strip()
        .rstrip(". ")
    )
    if cleaned in {"", ".", ".."}:
        cleaned = fallback
    if cleaned.split(".", maxsplit=1)[0].upper() in _WINDOWS_RESERVED_DEVICE_NAMES:
        cleaned = f"_{cleaned}"
    if len(cleaned) > max_length:
        digest = stable_hash(original)[:16]
        cleaned = f"{cleaned[: max_length - len(digest) - 1]}_{digest}"
    return cleaned


def _crop_suffix(image_id: str) -> str:
    suffix = Path(image_id).suffix.lower()
    return suffix if suffix in _SAFE_CROP_SUFFIXES else ".png"


def _crop_path(root: Path, candidate_id: str, crop: CropRecord) -> Path:
    image_token = _safe_name(Path(crop.image_id).stem, fallback="image")
    crop_token = _safe_name(crop.crop_id, fallback="crop")
    suffix = _crop_suffix(crop.image_id)
    return (
        root
        / "crops"
        / _safe_name(candidate_id, fallback="candidate")
        / (f"{image_token}__{crop_token}{suffix}")
    )


def materialize_crop(
    record: TaskRecord | InferenceRecord, crop: CropRecord, root: str | Path, candidate_id: str
) -> str:
    """Materialize an evidence crop outside the source-data archive for local SFT only.

    The function is intentionally invoked only by local commands. Generated crops are never committed to the
    repository and the manifest records their derived, non-redistributable nature.
    """
    out = _crop_path(Path(root), candidate_id, crop)
    if out.exists():
        try:
            with Image.open(out) as cached:
                cached.verify()
            return str(out)
        except Exception:
            out.unlink(missing_ok=True)
    ensure_dir(out.parent)
    with Image.open(record.image_path) as source:
        image = source.convert("RGB")
        box = crop.box_xyxy.clamp(record.width, record.height)
        left = int(max(0, round(box.x1)))
        top = int(max(0, round(box.y1)))
        right = int(min(record.width, max(left + 1, round(box.x2))))
        bottom = int(min(record.height, max(top + 1, round(box.y2))))
        rendered = image.crop((left, top, right, bottom))
        with tempfile.NamedTemporaryFile(
            prefix=f".{out.stem}.", suffix=out.suffix, dir=out.parent, delete=False
        ) as handle:
            temp = Path(handle.name)
        try:
            rendered.save(temp)
            with Image.open(temp) as verify_image:
                verify_image.verify()
            os.replace(temp, out)
        finally:
            temp.unlink(missing_ok=True)
    return str(out)


def _supporting_crop_ids(record: TaskRecord, crops: list[CropRecord]) -> list[str]:
    supports = [x for x in record.gt_objects if x.object_id in record.supporting_object_ids]
    if not supports:
        return []
    scored: list[tuple[float, str]] = []
    for crop in crops:
        score = max((box_iou(crop.box_xyxy, gt.box_xyxy) for gt in supports), default=0.0)
        if score > 0.0:
            scored.append((score, crop.crop_id))
    return [crop_id for _, crop_id in sorted(scored, reverse=True)]


def _prompt(record: TaskRecord, crop_ids: list[str]) -> str:
    crop_text = (
        "No evidence crops are supplied. Use the full scene only."
        if not crop_ids
        else "Evidence crops are provided after the full scene in this order: "
        + ", ".join(crop_ids)
        + "."
    )
    return (
        f"{record.prompt}\n{crop_text}\n"
        "Return one compact JSON object with keys: tool, decision, confidence, evidence_ids, reason_code. "
        "Use tool=FINALIZE. evidence_ids must refer only to supplied crop identifiers."
    )


def _target(record: TaskRecord, crop_ids: list[str]) -> str:
    return json.dumps(
        {
            "tool": "FINALIZE",
            "decision": record.action_label,
            "confidence": 0.95,
            "evidence_ids": crop_ids,
            "reason_code": "deterministic_label_policy_supervision",
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def build_sft_examples(
    records: list[TaskRecord],
    crops_by_image: dict[str, list[CropRecord]],
    candidate_id: str,
    split: str,
    output_dir: str | Path,
    max_crops: int = 3,
) -> tuple[list[SFTExample], dict[str, Any]]:
    """Build train/dev-only multimodal SFT examples from deterministic task labels.

    `final` is deliberately forbidden so frozen final records cannot reach SFT preparation. Ground truth is used
    only to construct supervised target text during train/dev preparation, never by model inference APIs.
    """
    if split not in {"train", "dev"}:
        raise ValueError(
            "SFT preparation is restricted to train or dev; calibration/final are not allowed"
        )
    root = ensure_dir(output_dir)
    chosen = [record for record in records if record.split == split]
    if not chosen:
        raise ValueError(f"No records found for split={split!r}")
    examples: list[SFTExample] = []
    crop_count = 0
    for record in chosen:
        selected = (
            []
            if candidate_id == "C2"
            else sorted(
                crops_by_image.get(record.image_id, []), key=lambda crop: crop.selection_rank
            )[:max_crops]
        )
        crop_paths = [materialize_crop(record, crop, root, candidate_id) for crop in selected]
        crop_ids = _supporting_crop_ids(record, selected)
        examples.append(
            SFTExample(
                example_id=f"{record.task_id}:{candidate_id}:{split}",
                image_id=record.image_id,
                split=split,
                candidate_id=candidate_id,
                image_paths=[record.image_path, *crop_paths],
                prompt=_prompt(record, [crop.crop_id for crop in selected]),
                target=_target(record, crop_ids),
                label_policy_version=record.label_policy_version,
            )
        )
        crop_count += len(selected)
    out_path = root / f"sft_{split}_{candidate_id}.jsonl"
    write_jsonl(out_path, [example.to_dict() for example in examples])
    manifest = {
        "status": "PASS",
        "purpose": "train/dev-only QLoRA supervised fine-tuning dataset preparation",
        "created_at_utc": utc_now(),
        "candidate_id": candidate_id,
        "split": split,
        "example_count": len(examples),
        "materialized_crop_count": crop_count,
        "max_crops": max_crops,
        "frozen_final_included": False,
        "label_policy": "deterministic task-policy labels used only as train/dev supervision",
        "claim_boundary": "SFT dataset preparation is not VLM training or performance validation.",
        "dataset_fingerprint": stable_hash([example.to_dict() for example in examples]),
        "output": str(out_path),
    }
    write_json(root / f"sft_{split}_{candidate_id}_manifest.json", manifest)
    return examples, manifest


def load_sft_examples(path: str | Path) -> list[SFTExample]:
    out: list[SFTExample] = []
    for row in read_jsonl(path):
        if row.get("split") not in {"train", "dev"}:
            raise ValueError("SFT dataset contains a forbidden non-train/dev split")
        out.append(
            SFTExample(
                example_id=str(row["example_id"]),
                image_id=str(row["image_id"]),
                split=str(row["split"]),
                candidate_id=str(row["candidate_id"]),
                image_paths=[str(x) for x in row["image_paths"]],
                prompt=str(row["prompt"]),
                target=str(row["target"]),
                label_policy_version=str(row["label_policy_version"]),
            )
        )
    return out


def prepare_sft_stage(
    records_path: str | Path,
    crops_path: str | Path | None,
    candidate_id: str,
    split: str,
    output: str | Path,
    max_crops: int = 3,
) -> dict[str, Any]:
    records = load_records(records_path)
    crops = load_crops(crops_path, candidate_id) if crops_path else {}
    _, manifest = build_sft_examples(
        records, crops, candidate_id, split, output, max_crops=max_crops
    )
    return manifest
