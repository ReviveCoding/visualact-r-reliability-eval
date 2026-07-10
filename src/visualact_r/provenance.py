"""Runtime provenance checks for source-vs-distribution validation.

The checks are intentionally small and dependency-free so CI and local verification can
confirm that wheel/sdist smoke runs execute the installed distribution rather than
accidentally importing the checkout's ``src`` tree.
"""

from __future__ import annotations

from pathlib import Path

import visualact_r


def runtime_provenance(source_root: str | Path) -> dict[str, str | bool]:
    """Describe whether the imported package resolves from a supplied checkout.

    ``source_root`` should be the repository root, not the ``src`` directory.
    The function resolves paths before comparison so symlinks cannot hide a source import.
    """
    root = Path(source_root).resolve()
    source_tree = (root / "src").resolve()
    module_file = Path(visualact_r.__file__ or "").resolve()
    try:
        imported_from_source_tree = module_file.is_relative_to(source_tree)
    except AttributeError:  # pragma: no cover - Python >=3.11 is required
        imported_from_source_tree = str(module_file).startswith(str(source_tree))
    return {
        "package_file": str(module_file),
        "source_tree": str(source_tree),
        "imported_from_source_tree": imported_from_source_tree,
    }


def assert_installed_distribution(source_root: str | Path) -> dict[str, str | bool]:
    """Fail closed when a packaged smoke test imports the checkout source tree."""
    result = runtime_provenance(source_root)
    if result["imported_from_source_tree"]:
        raise RuntimeError(
            "Packaged smoke test imported the checkout source tree instead of the installed "
            "wheel/sdist distribution. Run the command from a fresh distribution environment."
        )
    return result
