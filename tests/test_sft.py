from pathlib import Path

import pytest

from visualact_r.cef.core import fit_cef, make_crops
from visualact_r.agents.sft import build_sft_examples, load_sft_examples
from visualact_r.detector.synthetic import SyntheticColorDetector


def test_sft_preparation_excludes_final(records, config, tmp_path):
    detector = SyntheticColorDetector()
    cache = {record.image_id: detector.predict(record) for record in records}
    train = [record for record in records if record.split == "train"]
    cef = fit_cef(train, cache, config)
    crops = {}
    for record in records:
        raw, nms = cache[record.image_id]
        crops[record.image_id] = make_crops(record, raw, nms, "cef", config, cef)
    examples, manifest = build_sft_examples(
        records,
        crops,
        candidate_id="C5",
        split="train",
        output_dir=tmp_path / "sft",
    )
    assert manifest["frozen_final_included"] is False
    assert examples
    assert all(example.split == "train" for example in examples)
    assert all(Path(path).exists() for example in examples for path in example.image_paths)
    loaded = load_sft_examples(tmp_path / "sft" / "sft_train_C5.jsonl")
    assert len(loaded) == len(examples)
    with pytest.raises(ValueError, match="restricted"):
        build_sft_examples(records, crops, "C5", "final", tmp_path / "bad")


def test_sft_crop_paths_are_windows_portable(records, config, tmp_path):
    """Colon-delimited provenance IDs must not be emitted as Windows filenames."""
    from visualact_r.agents.sft import _crop_path, materialize_crop

    detector = SyntheticColorDetector()
    record = next(item for item in records if item.split == "train")
    raw, nms = detector.predict(record)
    crop = make_crops(record, raw, nms, "cef", config, None)[0]
    crop.crop_id = f"{record.image_id}:cef:{record.image_id}:cluster:0"

    target = _crop_path(tmp_path / "sft", "C5", crop)
    forbidden = set('<>:"/\\|?*')
    assert not (set(target.name) & forbidden)
    assert target == _crop_path(tmp_path / "sft", "C5", crop)

    materialized = Path(materialize_crop(record, crop, tmp_path / "sft", "C5"))
    assert materialized == target
    assert materialized.is_file()


def test_sft_filename_sanitizer_handles_windows_reserved_forms():
    from visualact_r.agents.sft import _safe_name

    sanitized = _safe_name('unsafe:<>"/\\|?*\x00. ')
    assert not (set(sanitized) & set('<>:"/\\|?*'))
    assert sanitized == sanitized.rstrip(". ")
    assert _safe_name("CON") == "_CON"
