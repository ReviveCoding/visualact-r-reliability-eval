from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from visualact_r.agents.qwen_adapter import QwenVLMAdapter, QwenVLMConfig, build_qlora_model
from visualact_r.utils import sha256_directory


class _ImageProcessor:
    def __init__(self) -> None:
        self.size = {"longest_edge": 999999, "shortest_edge": 12345}


class _Processor:
    def __init__(self) -> None:
        self.image_processor = _ImageProcessor()


def test_qwen_model_resolver_prefers_official_image_text_class() -> None:
    official = object()
    compatibility = object()
    module = SimpleNamespace(
        AutoModelForImageTextToText=official,
        AutoModelForMultimodalLM=compatibility,
    )
    assert QwenVLMAdapter._model_class(module) is official


def test_qwen_model_resolver_supports_compatibility_only_module() -> None:
    compatibility = object()
    module = SimpleNamespace(AutoModelForMultimodalLM=compatibility)
    assert QwenVLMAdapter._model_class(module) is compatibility


def test_qwen_model_resolver_fails_with_actionable_message() -> None:
    with pytest.raises(RuntimeError, match="transformers>=4.57.0"):
        QwenVLMAdapter._model_class(SimpleNamespace())


def test_qwen_image_budget_preserves_minimum_and_caps_longest_edge() -> None:
    processor = _Processor()
    QwenVLMAdapter._configure_image_budget(processor, 786432)
    assert processor.image_processor.size == {
        "longest_edge": 786432,
        "shortest_edge": 12345,
    }
    with pytest.raises(ValueError, match="positive"):
        QwenVLMAdapter._configure_image_budget(processor, 0)


def test_directory_fingerprint_covers_adapter_weights(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights-a")
    first = sha256_directory(adapter)
    (adapter / "adapter_model.safetensors").write_bytes(b"weights-b")
    assert sha256_directory(adapter) != first


def test_qlora_refuses_non_4bit_plan_before_optional_imports() -> None:
    config = QwenVLMConfig(model_id="Qwen/Qwen3-VL-4B-Instruct", load_in_4bit=False)
    with pytest.raises(ValueError, match="requires qwen.load_in_4bit=true"):
        build_qlora_model(config)


def test_qwen_image_budget_uses_max_pixels_when_processor_exposes_pixel_budget() -> None:
    processor = _Processor()
    processor.image_processor.size = {"min_pixels": 1024, "max_pixels": 999999}
    QwenVLMAdapter._configure_image_budget(processor, 262144)
    assert processor.image_processor.size == {"min_pixels": 1024, "max_pixels": 262144}
