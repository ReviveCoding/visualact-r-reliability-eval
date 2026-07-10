from __future__ import annotations

import json
import os
import shutil
import uuid
import zipfile
from pathlib import Path
from typing import Any

from visualact_r.utils import ensure_dir, sha256_file, utc_now, write_json


def _labels_from_zip(label_path: Path, target: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(label_path) as archive:
        json_names = [name for name in archive.namelist() if name.endswith(".json")]
        if len(json_names) != 1:
            raise ValueError(f"Expected one label JSON, found {json_names}")
        with archive.open(json_names[0]) as source, target.open("wb") as sink:
            shutil.copyfileobj(source, sink)
    with target.open("r", encoding="utf-8") as handle:
        labels = json.load(handle)
    if not isinstance(labels, list):
        raise ValueError("Extracted BDD label JSON must contain a top-level list")
    return labels


def _safe_member_names(archive: Any) -> list[str]:
    names = archive.getnames()
    if any(Path(name).name != name for name in names):
        raise ValueError("Nested/path-traversal archive name rejected")
    if len(names) != len(set(names)):
        raise ValueError("Duplicate file name within a BDD shard is not supported")
    return names


def extract_bdd_assets(
    data_root: str | Path,
    labels_zip: str,
    shards: list[str],
    output_root: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Extract user-supplied BDD assets through an isolated staging directory.

    The destination is published only after all archive members and the label/image join
    have been validated. Existing non-empty output is rejected unless ``overwrite=True``;
    that prevents a retry from silently mixing stale images from a prior archive set.
    """
    root = Path(data_root)
    destination = Path(output_root)
    label_path = root / labels_zip
    shard_paths = [root / shard_name for shard_name in shards]
    if not label_path.exists():
        raise FileNotFoundError(label_path)
    missing = [str(path) for path in shard_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing BDD shards: {missing}")
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Destination already contains files: {destination}. Use a new output path or --overwrite."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    labels_out = ensure_dir(staging / "labels")
    images_out = ensure_dir(staging / "images")
    shard_fingerprints: dict[str, str] = {}
    extracted_names: list[str] = []
    try:
        target = labels_out / "bdd100k_labels_images_val.json"
        annotations = _labels_from_zip(label_path, target)
        expected_names = [str(row.get("name", "")) for row in annotations]
        if not expected_names or any(not name for name in expected_names):
            raise ValueError("BDD labels contain missing image names")
        if len(expected_names) != len(set(expected_names)):
            raise ValueError("BDD labels contain duplicate image names")
        try:
            import py7zr
        except ImportError as exc:
            raise RuntimeError("py7zr is required for BDD 7z extraction") from exc
        for shard in shard_paths:
            with py7zr.SevenZipFile(shard, mode="r") as archive:
                names = _safe_member_names(archive)
                extracted_names.extend(names)
                archive.extractall(path=images_out)
            shard_fingerprints[shard.name] = sha256_file(shard)
        if len(extracted_names) != len(set(extracted_names)):
            raise ValueError(
                "Duplicate image names across BDD shards would overwrite extracted assets"
            )
        actual_names = sorted(path.name for path in images_out.iterdir() if path.is_file())
        if set(actual_names) != set(expected_names):
            missing_images = sorted(set(expected_names) - set(actual_names))[:25]
            extra_images = sorted(set(actual_names) - set(expected_names))[:25]
            raise RuntimeError(
                "Extracted BDD image/label join failed: "
                f"missing={missing_images}, extra={extra_images}"
            )
        manifest = {
            "status": "PASS",
            "created_at_utc": utc_now(),
            "labels_json": str(destination / "labels" / target.name),
            "images_dir": str(destination / "images"),
            "image_file_count": len(actual_names),
            "archive_member_count": len(extracted_names),
            "image_label_join_rate": 1.0,
            "source_fingerprints": {label_path.name: sha256_file(label_path), **shard_fingerprints},
            "publication": "staging_validated_then_published",
            "distribution_boundary": "External BDD data was extracted to a user-controlled path and must not be committed to Git.",
        }
        write_json(staging / "extract_manifest.json", manifest)
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staging, destination)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
