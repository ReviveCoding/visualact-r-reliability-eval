from __future__ import annotations

from pathlib import Path


def test_ci_covers_package_installs_and_both_hosted_os_paths() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "runs-on: ubuntu-latest" in workflow
    assert "runs-on: windows-latest" in workflow
    assert "workflow_dispatch:" in workflow
    assert "timeout-minutes:" in workflow
    assert "actions/checkout@v6" in workflow
    assert "actions/setup-python@v6" in workflow
    assert "actions/upload-artifact@v7" in workflow
    assert "python -m pip check" in workflow
    assert "python -m build --sdist --wheel --outdir dist" in workflow
    assert "wheel_fixture" in workflow
    assert "sdist_fixture" in workflow
    assert "System.Management.Automation.Language.Parser" in workflow
    assert "persist-credentials: false" in workflow
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD" in workflow
    assert "OPENBLAS_NUM_THREADS" in workflow
    assert "runtime-provenance" in workflow
