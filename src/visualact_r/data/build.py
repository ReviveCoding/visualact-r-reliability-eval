from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from visualact_r.data.task_policy import task_prompt, triage_policy
from visualact_r.schemas import Box, GroundTruthObject, TaskRecord
from visualact_r.utils import ensure_dir, read_json, stable_hash, write_json, write_jsonl


@lru_cache(maxsize=1)
def _dct_basis_32() -> np.ndarray:
    """Build the fixed orthonormal 32×32 DCT basis once per process."""
    n = 32
    indices: Any = np.arange(n, dtype=np.float32)
    basis = np.cos(np.pi * (2.0 * indices[None, :] + 1.0) * indices[:, None] / (2.0 * n))
    basis[0, :] *= 1.0 / np.sqrt(n)
    basis[1:, :] *= np.sqrt(2.0 / n)
    return basis


def perceptual_hash_image(image: Image.Image) -> str:
    """Compute the DCT pHash from an already-open image to avoid duplicate image I/O."""
    grayscale = image.convert("L")
    try:
        arr = np.asarray(grayscale.resize((32, 32)), dtype=np.float32)
    finally:
        grayscale.close()
    basis = _dct_basis_32()
    dct = basis @ arr @ basis.T
    low = dct[:8, :8].ravel()
    median = float(np.median(low[1:]))
    bits = (low >= median).astype(np.uint8)
    return f"{int(''.join(map(str, bits.tolist())), 2):016x}"


def perceptual_hash(path: str | Path) -> str:
    """Compute a compact DCT pHash used only for near-duplicate split grouping.

    This is deliberately independent of labels and detector outputs. The 8x8 low-frequency block gives a
    conventional perceptual-hash representation rather than the former average-hash approximation.
    """
    with Image.open(path) as image:
        return perceptual_hash_image(image)


# Compatibility alias for callers from the pre-v0.3 source tree. New reports call this DCT pHash, not aHash.
def average_hash(path: str | Path) -> str:
    return perceptual_hash(path)


def _annotation_objects(row: dict[str, Any]) -> list[GroundTruthObject]:
    out: list[GroundTruthObject] = []
    for label in row.get("labels", []):
        box = label.get("box2d")
        if not box:
            continue
        out.append(
            GroundTruthObject(
                object_id=f"{row['name']}:{label.get('id', len(out))}",
                category=str(label.get("category", "unknown")),
                box_xyxy=Box(
                    float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])
                ),
                occluded=bool(label.get("attributes", {}).get("occluded", False)),
                truncated=bool(label.get("attributes", {}).get("truncated", False)),
            )
        )
    return out


def build_task_records(
    annotation_path: str | Path,
    images_dir: str | Path,
    config: dict[str, Any],
    output_dir: str | Path,
) -> list[TaskRecord]:
    annotation_path, images_dir, output_dir = (
        Path(annotation_path),
        Path(images_dir),
        ensure_dir(output_dir),
    )
    annotations = read_json(annotation_path)
    rows: list[TaskRecord] = []
    for row in annotations:
        name = row["name"]
        image_path = images_dir / name
        if not image_path.exists():
            raise FileNotFoundError(f"Missing extracted image for task build: {image_path}")
        with Image.open(image_path) as image:
            width, height = image.size
            image_phash = perceptual_hash_image(image)
        objects = _annotation_objects(row)
        action, supporting, flags = triage_policy(objects, width, height, config)
        attrs = row.get("attributes", {})
        task_id = f"{Path(name).stem}:triage_standard"
        rows.append(
            TaskRecord(
                image_id=name,
                image_path=str(image_path),
                width=width,
                height=height,
                task_id=task_id,
                task_template_id="triage_standard",
                prompt=task_prompt("triage_standard"),
                weather=str(attrs.get("weather", "unknown")),
                scene=str(attrs.get("scene", "unknown")),
                timeofday=str(attrs.get("timeofday", "unknown")),
                gt_objects=objects,
                action_label=action,
                supporting_object_ids=supporting,
                hard_flags=flags,
                phash=image_phash,
                label_policy_version=config["project"]["task_policy_version"],
            )
        )
    write_jsonl(output_dir / "task_records_unsplit.jsonl", [x.to_dict() for x in rows])
    write_json(
        output_dir / "task_build_manifest.json",
        {
            "record_count": len(rows),
            "task_policy_version": config["project"]["task_policy_version"],
            "input_annotations": str(annotation_path),
            "images_dir": str(images_dir),
            "config_hash": stable_hash(config),
            "image_hash_method": "dct_phash_8x8_from_32x32_grayscale",
        },
    )
    return rows
