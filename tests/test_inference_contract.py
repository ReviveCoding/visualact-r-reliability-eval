from __future__ import annotations

from pathlib import Path

from visualact_r.artifacts import load_inference_records
from visualact_r.data.splits import build_splits
from visualact_r.utils import read_jsonl


def test_build_splits_writes_label_free_inference_contract(records, config, tmp_path: Path) -> None:
    root = tmp_path / "rebuilt"
    built = build_splits(records, config, str(root))
    inference_path = root / "inference_records.jsonl"
    rows = read_jsonl(inference_path)
    assert len(rows) == len(built)
    forbidden = {
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
    assert all(not (forbidden & set(row)) for row in rows)
    loaded = load_inference_records(inference_path, "final")
    assert all(row.split == "final" for row in loaded)
