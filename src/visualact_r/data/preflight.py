from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image

from visualact_r.utils import ensure_dir, sha256_file, utc_now, write_json


def _load_annotations(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [x for x in archive.namelist() if x.endswith(".json")]
            if len(names) != 1:
                raise ValueError(f"Expected exactly one JSON in {path}, found {names}")
            with archive.open(names[0]) as handle:
                return json.load(handle)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _archive_names(path: Path) -> list[str]:
    try:
        import py7zr
    except ImportError as exc:
        raise RuntimeError("py7zr is required to inspect external 7z BDD shards") from exc
    with py7zr.SevenZipFile(path, mode="r") as archive:
        return [Path(x).name for x in archive.getnames()]


BOX_REQUIRED_CATEGORIES = {
    "person",
    "rider",
    "bike",
    "motor",
    "car",
    "truck",
    "bus",
    "traffic light",
    "traffic sign",
    "train",
}


def _validate_annotation_row(row: dict[str, Any]) -> list[str]:
    """Accept BDD's mixed box and poly2d labels while validating all box-required categories."""
    problems: list[str] = []
    if not isinstance(row.get("name"), str):
        problems.append("missing_name")
    if not isinstance(row.get("labels", []), list):
        problems.append("labels_not_list")
    for item in row.get("labels", []):
        category = str(item.get("category", ""))
        box = item.get("box2d")
        if category in BOX_REQUIRED_CATEGORIES and (
            not isinstance(box, dict) or not all(k in box for k in ("x1", "y1", "x2", "y2"))
        ):
            problems.append("invalid_box_required_category")
        if isinstance(box, dict) and not all(k in box for k in ("x1", "y1", "x2", "y2")):
            problems.append("invalid_box")
    return problems


def preflight_external_bdd(
    data_root: str | Path,
    labels_zip: str,
    shards: list[str],
    output_dir: str | Path,
) -> dict[str, Any]:
    root = Path(data_root)
    label_path = root / labels_zip
    shard_paths = [root / x for x in shards]
    if not label_path.exists():
        raise FileNotFoundError(label_path)
    missing = [str(x) for x in shard_paths if not x.exists()]
    if missing:
        raise FileNotFoundError(f"Missing BDD shards: {missing}")
    annotations = _load_annotations(label_path)
    names = [str(row.get("name", "")) for row in annotations]
    archive_names: list[str] = []
    for shard in shard_paths:
        archive_names.extend(_archive_names(shard))
    annotation_set = set(names)
    archive_set = set(archive_names)
    invalid_rows = sum(bool(_validate_annotation_row(row)) for row in annotations)
    box_label_count = sum(
        1
        for row in annotations
        for item in row.get("labels", [])
        if isinstance(item.get("box2d"), dict)
    )
    poly_or_other_label_count = sum(
        1
        for row in annotations
        for item in row.get("labels", [])
        if not isinstance(item.get("box2d"), dict)
    )
    duplicate_annotation_names = len(names) - len(annotation_set)
    duplicate_archive_names = len(archive_names) - len(archive_set)
    intersection = annotation_set & archive_set
    join_rate = len(intersection) / len(annotation_set) if annotation_set else 0.0
    report = {
        "status": "PASS"
        if join_rate == 1.0
        and invalid_rows == 0
        and duplicate_annotation_names == 0
        and duplicate_archive_names == 0
        else "HOLD",
        "purpose": "external BDD archive and annotation preflight",
        "created_at_utc": utc_now(),
        "annotation_count": len(annotations),
        "archive_image_count": len(archive_set),
        "image_label_join_rate": join_rate,
        "image_label_intersection_count": len(intersection),
        "annotation_missing_images": sorted(annotation_set - archive_set)[:25],
        "archive_images_without_annotations": sorted(archive_set - annotation_set)[:25],
        "invalid_annotation_rows": invalid_rows,
        "box_label_count": box_label_count,
        "poly_or_other_label_count": poly_or_other_label_count,
        "duplicate_annotation_names": duplicate_annotation_names,
        "duplicate_archive_names": duplicate_archive_names,
        "source_fingerprints": {
            str(label_path.name): sha256_file(label_path),
            **{x.name: sha256_file(x) for x in shard_paths},
        },
        "next_step": "build-data requires extracted images_dir for pHash/image validation"
        if join_rate == 1.0
        else "fix source asset mismatch before any split or training",
    }
    output_dir = ensure_dir(output_dir)
    write_json(output_dir / "preflight_data_report.json", report)
    return report


def preflight_fixture(fixture_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    root = Path(fixture_root)
    ann = root / "fixture_annotations.json"
    images = root / "images"
    annotations = _load_annotations(ann)
    names = [row["name"] for row in annotations]
    image_names = [x.name for x in images.glob("*.png")]
    valid_image_sample = 0
    for image_path in sorted(images.glob("*.png"))[:5]:
        with Image.open(image_path) as image:
            image.verify()
        valid_image_sample += 1
    report = {
        "status": "PASS" if set(names) == set(image_names) else "HOLD",
        "purpose": "synthetic fixture preflight",
        "created_at_utc": utc_now(),
        "annotation_count": len(names),
        "archive_image_count": len(image_names),
        "image_label_join_rate": len(set(names) & set(image_names)) / len(set(names)),
        "sample_images_verified": valid_image_sample,
        "gt_fields_in_inference_contract": 0,
    }
    output_dir = ensure_dir(output_dir)
    write_json(output_dir / "preflight_data_report.json", report)
    return report
