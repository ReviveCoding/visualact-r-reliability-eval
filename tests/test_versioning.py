from __future__ import annotations

import tomllib
from pathlib import Path

from visualact_r import __version__


def test_runtime_version_matches_pyproject() -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert __version__ == project["project"]["version"]
