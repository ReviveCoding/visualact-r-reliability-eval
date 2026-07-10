from __future__ import annotations

from pathlib import Path

import pytest

from visualact_r.agents.feasibility import LocalQLoRAFeasibilityConfig


def test_feasibility_config_rejects_missing_image(tmp_path: Path) -> None:
    config = LocalQLoRAFeasibilityConfig(image_path=str(tmp_path / "missing.png"))
    with pytest.raises(FileNotFoundError, match="Feasibility image"):
        config.validate()


def test_feasibility_config_rejects_invalid_budget(tmp_path: Path) -> None:
    image = tmp_path / "scene.png"
    image.write_bytes(b"fixture")
    config = LocalQLoRAFeasibilityConfig(image_path=str(image), memory_budget_fraction=1.1)
    with pytest.raises(ValueError, match="memory_budget_fraction"):
        config.validate()
