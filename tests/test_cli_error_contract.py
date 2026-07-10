from __future__ import annotations

from pathlib import Path

import pytest

from visualact_r.cli import main


def test_cli_returns_clean_operational_error_for_existing_config_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    existing = tmp_path / "existing.yaml"
    existing.write_text("already here\n", encoding="utf-8")
    with pytest.raises(SystemExit) as result:
        main(["write-config", "--output", str(existing)])
    captured = capsys.readouterr()
    assert result.value.code == 2
    assert "ERROR [write-config]" in captured.err
    assert "Refusing to overwrite existing config" in captured.err
