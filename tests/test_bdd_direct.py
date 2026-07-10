from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from visualact_r.artifacts import load_inference_records, load_records
from visualact_r.config import load_config
from visualact_r.data.bdd_direct import (
    BDDDirectLayoutError,
    build_direct_bdd_records,
    inspect_extracted_bdd_layout,
)
from visualact_r.utils import read_json


def _row(name: str) -> dict[str, object]:
    return {
        "name": name,
        "attributes": {"weather": "clear", "scene": "city street", "timeofday": "daytime"},
        "labels": [
            {
                "id": "person-0",
                "category": "person",
                "attributes": {"occluded": False, "truncated": False},
                "box2d": {"x1": 4, "y1": 4, "x2": 16, "y2": 20},
            }
        ],
    }


def _make_image(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)


def _make_extracted_bdd(root: Path) -> Path:
    for split, count, start in (("train", 9, 1), ("val", 3, 101)):
        image_dir = root / "Images" / split
        label_dir = root / "Labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for index in range(count):
            name = f"{split}_{index:03d}.png"
            _make_image(image_dir / name, start + index)
            (label_dir / f"{name}.json").write_text(json.dumps(_row(name)), encoding="utf-8")
    return root


def test_direct_audit_and_build_preserve_bdd_val_as_final(tmp_path: Path) -> None:
    bdd = _make_extracted_bdd(tmp_path / "bdd")
    audit = inspect_extracted_bdd_layout(bdd)
    assert audit["status"] == "PASS"
    assert audit["train"]["annotation_row_count"] == 9
    assert audit["final"]["annotation_row_count"] == 3
    output = tmp_path / "output"
    report = build_direct_bdd_records(
        bdd,
        output,
        load_config(),
        cross_partition_phash_threshold=0,
    )
    assert report["status"] == "PASS"
    assert report["record_counts"]["final"] == 3
    records = load_records(output / "task_records.jsonl")
    assert {record.split for record in records if record.image_id.startswith("val_")} == {"final"}
    assert all(
        record.split in {"train", "dev", "calibration"}
        for record in records
        if record.image_id.startswith("train_")
    )
    inference = load_inference_records(output / "inference_records.jsonl")
    assert len(inference) == len(records)
    manifest = read_json(output / "direct_bdd_split_manifest.json")
    assert manifest["protocol"]["development_source"] == "BDD100K train only"
    assert manifest["protocol"]["frozen_final_source"] == "BDD100K val only"


def test_direct_audit_rejects_cross_source_exact_image_name_overlap(tmp_path: Path) -> None:
    bdd = _make_extracted_bdd(tmp_path / "bdd")
    source = bdd / "Images" / "train" / "train_000.png"
    target = bdd / "Images" / "val" / "train_000.png"
    target.write_bytes(source.read_bytes())
    (bdd / "Labels" / "val" / "train_000.png.json").write_text(
        json.dumps(_row("train_000.png")), encoding="utf-8"
    )
    audit = inspect_extracted_bdd_layout(bdd)
    assert audit["status"] == "HOLD"
    assert audit["cross_source_exact_image_name_overlap_count"] == 1


def test_direct_build_requires_positive_development_cap(tmp_path: Path) -> None:
    bdd = _make_extracted_bdd(tmp_path / "bdd")
    with pytest.raises(ValueError, match="max_train_records"):
        build_direct_bdd_records(bdd, tmp_path / "out", load_config(), max_train_records=0)


def test_direct_audit_rejects_row_without_name_when_filename_cannot_infer(tmp_path: Path) -> None:
    bdd = _make_extracted_bdd(tmp_path / "bdd")
    bad = bdd / "Labels" / "train" / "unrelated.json"
    bad.write_text(json.dumps({"labels": []}), encoding="utf-8")
    with pytest.raises(BDDDirectLayoutError, match="no image name"):
        inspect_extracted_bdd_layout(bdd)


def test_direct_audit_infers_image_name_from_stem_only_label_file(tmp_path: Path) -> None:
    bdd = _make_extracted_bdd(tmp_path / "bdd")
    label = bdd / "Labels" / "train" / "train_000.png.json"
    label.unlink()
    (bdd / "Labels" / "train" / "train_000.json").write_text(
        json.dumps({"labels": []}), encoding="utf-8"
    )
    audit = inspect_extracted_bdd_layout(bdd)
    assert audit["status"] == "PASS"
    assert audit["train"]["missing_image_count"] == 0


def test_direct_audit_rejects_ambiguous_stem_only_label_file(tmp_path: Path) -> None:
    bdd = _make_extracted_bdd(tmp_path / "bdd")
    image_dir = bdd / "Images" / "train"
    _make_image(image_dir / "ambiguous.png", 501)
    _make_image(image_dir / "ambiguous.jpg", 502)
    (bdd / "Labels" / "train" / "ambiguous.json").write_text(
        json.dumps({"labels": []}), encoding="utf-8"
    )
    with pytest.raises(BDDDirectLayoutError, match="maps to multiple images"):
        inspect_extracted_bdd_layout(bdd)


def test_direct_audit_cli_writes_hold_report_for_schema_error(tmp_path: Path) -> None:
    from argparse import Namespace

    from visualact_r.cli import cmd_bdd_direct_audit

    bdd = _make_extracted_bdd(tmp_path / "bdd")
    (bdd / "Labels" / "train" / "unmatched.json").write_text(
        json.dumps({"labels": []}), encoding="utf-8"
    )
    output = tmp_path / "audit.json"
    exit_code = cmd_bdd_direct_audit(Namespace(bdd_root=str(bdd), output=str(output)))
    report = read_json(output)
    assert exit_code == 2
    assert report["status"] == "HOLD"
    assert report["error_type"] == "BDDDirectLayoutError"


def test_direct_build_writes_hold_manifest_for_cross_partition_phash(tmp_path: Path) -> None:
    bdd = _make_extracted_bdd(tmp_path / "bdd")
    # Copy a train image into the BDD val source under a different name, preserving an exact visual duplicate.
    source = bdd / "Images" / "train" / "train_000.png"
    duplicate_name = "val_duplicate.png"
    (bdd / "Images" / "val" / duplicate_name).write_bytes(source.read_bytes())
    (bdd / "Labels" / "val" / f"{duplicate_name}.json").write_text(
        json.dumps(_row(duplicate_name)), encoding="utf-8"
    )
    output = tmp_path / "output"
    report = build_direct_bdd_records(
        bdd,
        output,
        load_config(),
        cross_partition_phash_threshold=0,
    )
    assert report["status"] == "HOLD"
    assert report["cross_partition_phash"]["match_count"] >= 1
    manifest = read_json(output / "direct_bdd_split_manifest.json")
    assert manifest["status"] == "HOLD"
    assert manifest["task_records_path"] is None
    assert not (output / "task_records.jsonl").exists()


def test_direct_build_cpu_phash_backend_records_manifest(tmp_path: Path) -> None:
    bdd = _make_extracted_bdd(tmp_path / "bdd")
    output = tmp_path / "output_cpu_backend"
    report = build_direct_bdd_records(
        bdd,
        output,
        load_config(),
        cross_partition_phash_threshold=0,
        phash_compare_backend="cpu",
        phash_chunk_size=2,
    )
    assert report["status"] == "PASS"
    assert report["cross_partition_phash"]["compare_backend_requested"] == "cpu"
    assert report["cross_partition_phash"]["compare_backend_used"] == "cpu"
    assert report["cross_partition_phash"]["chunk_size"] == 2


def test_direct_build_auto_phash_backend_records_backend(tmp_path: Path) -> None:
    bdd = _make_extracted_bdd(tmp_path / "bdd")
    output = tmp_path / "output_auto_backend"
    report = build_direct_bdd_records(
        bdd,
        output,
        load_config(),
        cross_partition_phash_threshold=0,
        phash_compare_backend="auto",
        phash_chunk_size=2,
    )
    assert report["status"] == "PASS"
    assert report["cross_partition_phash"]["compare_backend_requested"] == "auto"
    assert report["cross_partition_phash"]["compare_backend_used"] in {"cpu", "cuda"}
    if report["cross_partition_phash"]["compare_backend_used"] == "cuda":
        assert report["cross_partition_phash"]["cuda"] is not None


def test_cuda_phash_backend_matches_cpu_when_cuda_is_available(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA pHash backend requires a CUDA GPU")
    bdd = _make_extracted_bdd(tmp_path / "bdd")
    config = load_config()
    cpu = build_direct_bdd_records(
        bdd,
        tmp_path / "cpu",
        config,
        cross_partition_phash_threshold=0,
        phash_compare_backend="cpu",
    )
    cuda = build_direct_bdd_records(
        bdd,
        tmp_path / "cuda",
        config,
        cross_partition_phash_threshold=0,
        phash_compare_backend="cuda",
        phash_chunk_size=2,
    )
    assert (
        cpu["cross_partition_phash"]["match_count"] == cuda["cross_partition_phash"]["match_count"]
    )
    assert cpu["cross_partition_phash"]["examples"] == cuda["cross_partition_phash"]["examples"]
    assert cuda["cross_partition_phash"]["compare_backend_used"] == "cuda"
    assert cuda["cross_partition_phash"]["cuda"]["cuda_peak_memory_reserved_bytes"] >= 0
