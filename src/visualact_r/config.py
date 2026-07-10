from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from visualact_r.utils import deep_merge, load_yaml

_TEMPLATE_FILES = {
    "default": "default.yaml",
    "qwen_qlora": "qwen_qlora.yaml",
    "local_qlora_pilot": "local_qlora_pilot.yaml",
}


def _resource_text(name: str) -> str:
    """Read a package-bundled YAML template so wheel and sdist installs need no source checkout."""
    return resources.files("visualact_r.resources").joinpath(name).read_text(encoding="utf-8")


def config_template_names() -> tuple[str, ...]:
    """Return stable CLI-visible template names."""
    return tuple(_TEMPLATE_FILES)


def write_config_template(
    destination: str | Path, template: str = "default", *, overwrite: bool = False
) -> Path:
    """Materialize a bundled template to a user-controlled editable file.

    Packaged resources are canonical. This avoids drift from keeping duplicated source-tree
    copies of the same YAML files while retaining an explicit editable-config workflow.
    """
    if template not in _TEMPLATE_FILES:
        raise ValueError(f"Unknown bundled config template: {template!r}")
    path = Path(destination)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing config: {path}. Use --force after reviewing it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_resource_text(_TEMPLATE_FILES[template]), encoding="utf-8")
    return path


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the canonical bundled default config and optionally merge a user override file."""
    base = yaml.safe_load(_resource_text(_TEMPLATE_FILES["default"])) or {}
    if path is None:
        return base
    return deep_merge(base, load_yaml(path))
