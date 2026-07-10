from __future__ import annotations

from pathlib import Path

import pytest

from visualact_r.provenance import assert_installed_distribution, runtime_provenance


def test_runtime_provenance_detects_checkout_source_tree() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = runtime_provenance(repo_root)
    assert result["imported_from_source_tree"] is True
    with pytest.raises(RuntimeError, match="checkout source tree"):
        assert_installed_distribution(repo_root)


def test_runtime_provenance_accepts_unrelated_root(tmp_path: Path) -> None:
    result = assert_installed_distribution(tmp_path)
    assert result["imported_from_source_tree"] is False
    assert Path(str(result["package_file"])).exists()
