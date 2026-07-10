from __future__ import annotations

from pathlib import Path

import pytest

from visualact_r.stages import _validate_real_qwen_rollout_protocol
from visualact_r.utils import sha256_directory, sha256_file, write_json, write_jsonl


def _build_protocol_fixture(tmp_path: Path) -> tuple[dict[str, str], dict[str, str], Path]:
    data = tmp_path / "data"
    data.mkdir()
    records = data / "task_records.jsonl"
    inference = data / "inference_records.jsonl"
    write_jsonl(records, [{"image_id": "i", "split": "final"}])
    write_jsonl(
        inference,
        [
            {
                "image_id": "i",
                "image_path": "x.png",
                "width": 10,
                "height": 10,
                "task_id": "t",
                "task_template_id": "v",
                "prompt": "p",
                "split": "final",
                "label_policy_version": "v1",
            }
        ],
    )
    adapter = tmp_path / "qwen_shared" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    fingerprint = sha256_directory(adapter)
    write_json(
        adapter.parent / "qwen_qlora_training_manifest.json",
        {
            "status": "TRAINED_LOCAL_GPU",
            "training_protocol": "shared_full_scene_only",
            "adapter_fingerprint": fingerprint,
        },
    )
    traces: dict[str, str] = {}
    crops: dict[str, str] = {}
    for candidate in ("C3", "C4", "C5"):
        root = tmp_path / candidate
        root.mkdir()
        trace = root / f"agent_traces_{candidate}.jsonl"
        crop = root / f"crop_records_{candidate}.jsonl"
        write_jsonl(trace, [])
        write_jsonl(crop, [{"candidate_id": candidate, "image_id": "i"}])
        write_json(
            root / f"rollout_{candidate}_report.json",
            {
                "status": "PASS",
                "candidate_id": candidate,
                "split": "final",
                "label_fields_read": 0,
                "ground_truth_fields_read": 0,
                "source_inference_records_sha256": sha256_file(inference),
                "source_crops_sha256": sha256_file(crop),
                "adapter_fingerprint": fingerprint,
                "adapter_dir": str(adapter),
                "base_model": "Qwen/Qwen3-VL-4B-Instruct",
                "revision": "main",
                "quantization": {"load_in_4bit": True, "quant_type": "nf4"},
                "max_image_pixels": 786432,
                "config_hash": "same-config",
            },
        )
        traces[candidate] = str(trace)
        crops[candidate] = str(crop)
    return traces, crops, records


def test_qwen_real_protocol_binds_inference_crops_and_full_adapter(tmp_path: Path) -> None:
    traces, crops, records = _build_protocol_fixture(tmp_path)
    report = _validate_real_qwen_rollout_protocol(traces, crops, str(records))
    assert report["shared_adapter_fingerprint"]
    assert report["source_inference_records_sha256"]


def test_qwen_real_protocol_rejects_stale_crop_report(tmp_path: Path) -> None:
    traces, crops, records = _build_protocol_fixture(tmp_path)
    Path(crops["C5"]).write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="Invalid sealed Qwen rollout lineage"):
        _validate_real_qwen_rollout_protocol(traces, crops, str(records))
