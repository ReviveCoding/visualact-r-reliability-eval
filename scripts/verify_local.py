"""Cross-platform local verification for the CPU-contract release path.

The verifier stages a clean copy of the repository beneath its output directory. That prevents
editable installs, build metadata, bytecode caches, or generated fixture artifacts from mutating
the developer checkout. It validates source quality, synthetic pipeline connectivity, wheel and
sdist installation, and package-resource loading. It deliberately does not download BDD media or
Qwen weights.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


_EXCLUDED_COPY_NAMES = {
    ".git",
    ".venv",
    ".wheel-venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "build",
    "dist",
    "runs",
    "external_data",
    "models",
    "checkpoints",
    "__pycache__",
}


def _run(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    environment = os.environ.copy()
    # Fixture validation uses small numerical workloads. Pinning thread pools prevents
    # unrelated BLAS defaults or preloaded pytest plugins from making this contract path
    # flaky or disproportionately slow on shared local/hosted runners.
    environment.setdefault("PYTHONUTF8", "1")
    environment.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    environment.setdefault("NUMEXPR_NUM_THREADS", "1")
    environment.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    environment.setdefault("PIP_NO_INPUT", "1")
    subprocess.run(command, cwd=cwd, check=True, env=environment)


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if sys.platform.startswith("win") else "bin/python")


def _fresh_venv(python: str, path: Path, cwd: Path) -> Path:
    _run([python, "-m", "venv", str(path)], cwd)
    executable = _venv_python(path)
    _run([str(executable), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], cwd)
    return executable


def _ignore_copy(_: str, names: list[str]) -> set[str]:
    ignored = set(_EXCLUDED_COPY_NAMES & set(names))
    ignored.update(name for name in names if name.endswith(".egg-info"))
    return ignored


def _stage_source(repo: Path, output: Path) -> Path:
    staged = output / "source_tree"
    shutil.copytree(repo, staged, ignore=_ignore_copy)
    return staged


def _run_contract(python: Path, cwd: Path, run_dir: Path) -> None:
    _run([str(python), "-m", "visualact_r", "--help"], cwd)
    _run([str(python), "-m", "visualact_r", "fixture-run", "--output", str(run_dir)], cwd)
    _run([str(python), "-m", "visualact_r", "qualify", "--run-dir", str(run_dir)], cwd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("runs/local_verify"))
    parser.add_argument(
        "--keep", action="store_true", help="Keep staged source and virtual environments"
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    output = (
        (repo / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    )
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    # Reuse downloaded wheels across the source, wheel, and sdist environments.
    # By default the cache is disposable under the verification output; a caller may
    # set VISUALACT_R_PIP_CACHE to reuse a trusted external pip cache across runs.
    cache_dir = Path(os.environ.get("VISUALACT_R_PIP_CACHE", output / "pip_cache")).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PIP_CACHE_DIR"] = str(cache_dir)
    os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    find_links = os.environ.get("VISUALACT_R_PIP_FIND_LINKS")
    if find_links:
        wheelhouse = Path(find_links).resolve()
        if not wheelhouse.is_dir():
            raise ValueError(f"VISUALACT_R_PIP_FIND_LINKS is not a directory: {wheelhouse}")
        # Optional offline mode for repeatable, bandwidth-efficient clean verification.
        os.environ["PIP_FIND_LINKS"] = str(wheelhouse)
        os.environ["PIP_NO_INDEX"] = "1"
    staged = _stage_source(repo, output)

    source_python = _fresh_venv(sys.executable, output / "source_venv", staged)
    _run([str(source_python), "-m", "pip", "install", "-e", ".[dev]"], staged)
    _run([str(source_python), "-m", "pip", "check"], staged)
    _run([str(source_python), "-m", "compileall", "-q", "src"], staged)
    _run([str(source_python), "-m", "ruff", "check", "."], staged)
    _run([str(source_python), "-m", "ruff", "format", "--check", "."], staged)
    _run(
        [str(source_python), "-m", "mypy", "--python-version", "3.11", "--no-site-packages", "src"],
        staged,
    )
    _run([str(source_python), "-m", "pytest", "-q"], staged)
    _run_contract(source_python, staged, output / "fixture")

    dist = output / "dist"
    _run([str(source_python), "-m", "build", "--sdist", "--wheel", "--outdir", str(dist)], staged)
    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise RuntimeError(
            f"Expected one wheel and one sdist, found wheels={wheels}, sdists={sdists}"
        )

    wheel_python = _fresh_venv(sys.executable, output / "wheel_venv", staged)
    _run([str(wheel_python), "-m", "pip", "install", str(wheels[0])], staged)
    _run([str(wheel_python), "-m", "pip", "check"], staged)
    _run(
        [
            str(wheel_python),
            "-m",
            "visualact_r",
            "runtime-provenance",
            "--source-root",
            str(staged),
            "--output",
            str(output / "wheel_runtime_provenance.json"),
        ],
        staged,
    )
    _run_contract(wheel_python, staged, output / "wheel_fixture")
    _run(
        [
            str(wheel_python),
            "-m",
            "visualact_r",
            "write-config",
            "--template",
            "qwen_qlora",
            "--output",
            str(output / "qwen_qlora.yaml"),
        ],
        staged,
    )

    sdist_python = _fresh_venv(sys.executable, output / "sdist_venv", staged)
    _run([str(sdist_python), "-m", "pip", "install", str(sdists[0])], staged)
    _run([str(sdist_python), "-m", "pip", "check"], staged)
    _run(
        [
            str(sdist_python),
            "-m",
            "visualact_r",
            "runtime-provenance",
            "--source-root",
            str(staged),
            "--output",
            str(output / "sdist_runtime_provenance.json"),
        ],
        staged,
    )
    _run_contract(sdist_python, staged, output / "sdist_fixture")

    print(f"PASS: staged source, fixture, wheel, and sdist verification completed at {output}")
    if not args.keep:
        for directory in (output / "source_venv", output / "wheel_venv", output / "sdist_venv"):
            shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    main()
