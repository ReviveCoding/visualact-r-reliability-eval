from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from visualact_r.cef.core import _aggregate_evidence
from visualact_r.config import load_config, write_config_template
from visualact_r.data.build import perceptual_hash, perceptual_hash_image
from visualact_r.data.extract import extract_bdd_assets
from visualact_r.schemas import Box, CropRecord, GroundTruthObject


def _crop(crop_id: str, box: Box) -> CropRecord:
    return CropRecord(
        crop_id=crop_id,
        image_id="image.jpg",
        query_id="triage",
        box_xyxy=box,
        primary_candidate_id=crop_id,
        raw_candidate_ids=[crop_id],
        category="car",
        base_confidence=0.9,
        edge_expansion=[0.0] * 4,
        edge_deflation=[0.0] * 4,
        coverage_proxy=0.9,
        contamination_proxy=0.0,
        selection_rank=1,
        variant="cef",
    )


def test_cef_development_evidence_uses_union_area() -> None:
    support = [GroundTruthObject("support", "car", Box(0, 0, 10, 10))]
    crops = [_crop("left", Box(0, 0, 6, 10)), _crop("right", Box(4, 0, 14, 10))]
    metric = _aggregate_evidence(crops, support)
    assert metric["recall"] == 1.0
    assert metric["contamination"] == pytest.approx(2.0 / 7.0)


def test_config_writer_uses_bundled_template_and_refuses_accidental_overwrite(
    tmp_path: Path,
) -> None:
    target = tmp_path / "configs" / "qwen.local.yaml"
    written = write_config_template(target, "qwen_qlora")
    assert written == target
    assert "Qwen/Qwen3-VL-4B-Instruct" in target.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_config_template(target, "qwen_qlora")
    write_config_template(target, "default", overwrite=True)
    assert load_config(target)["project"]["name"] == "visualact-r"


def test_build_reuses_open_image_hash_semantics(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (24, 16), (20, 50, 90)).save(image_path)
    with Image.open(image_path) as image:
        assert perceptual_hash_image(image) == perceptual_hash(image_path)


def test_atomic_small_archive_extraction_and_overwrite_guard(tmp_path: Path) -> None:
    py7zr = pytest.importorskip("py7zr")
    source = tmp_path / "source"
    source.mkdir()
    image_a = source / "a.jpg"
    image_b = source / "b.jpg"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(image_a)
    Image.new("RGB", (8, 8), (40, 50, 60)).save(image_b)
    labels = [
        {"name": "a.jpg", "labels": []},
        {"name": "b.jpg", "labels": []},
    ]
    with zipfile.ZipFile(source / "labels.zip", "w") as archive:
        archive.writestr("bdd100k_labels_images_val.json", json.dumps(labels))
    for archive_name, image in [("one.7z", image_a), ("two.7z", image_b)]:
        with py7zr.SevenZipFile(source / archive_name, "w") as archive:
            archive.write(image, image.name)

    target = tmp_path / "cache"
    report = extract_bdd_assets(source, "labels.zip", ["one.7z", "two.7z"], target)
    assert report["status"] == "PASS"
    assert report["publication"] == "staging_validated_then_published"
    assert sorted(path.name for path in (target / "images").iterdir()) == ["a.jpg", "b.jpg"]
    with pytest.raises(FileExistsError):
        extract_bdd_assets(source, "labels.zip", ["one.7z", "two.7z"], target)
    assert (
        extract_bdd_assets(source, "labels.zip", ["one.7z", "two.7z"], target, overwrite=True)[
            "status"
        ]
        == "PASS"
    )


def test_powershell_preflight_starts_with_param_block_and_propagates_cli_failure() -> None:
    script = Path("scripts/run_local_bdd_preflight.ps1").read_text(encoding="utf-8").lstrip()
    assert script.startswith("param(")
    assert "if ($LASTEXITCODE -ne 0)" in script
    assert 'throw "VisualAct-R BDD preflight failed' in script
    assert "BDD_PREFLIGHT_WRAPPER=PASS" in script
