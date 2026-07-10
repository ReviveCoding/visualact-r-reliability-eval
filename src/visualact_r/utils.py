from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, cast

import numpy as np
import yaml


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def json_default(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(cast(Any, obj))
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)!r}")


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, dir=path.parent
    ) as handle:
        handle.write(text)
        temp_name = handle.name
    os.replace(temp_name, path)


def atomic_write_bytes(path: str | Path, content: bytes) -> None:
    """Atomically publish bytes so a failed serialization cannot leave a partial artifact."""
    path = Path(path)
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=path.parent) as handle:
        handle.write(content)
        temp_name = handle.name
    os.replace(temp_name, path)


def write_json(path: str | Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=json_default))


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path: str | Path, records: Iterable[Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, dir=path.parent
    ) as handle:
        for record in records:
            handle.write(json.dumps(record, default=json_default, sort_keys=True) + "\n")
        temp_name = handle.name
    os.replace(temp_name, path)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, default=json_default, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: str | Path) -> str:
    """Fingerprint all files beneath an artifact directory, including relative paths.

    Adapter configuration alone is insufficient provenance because weights can change while
    adapter_config.json remains identical. Symlinks are deliberately rejected to keep the
    digest portable and deterministic.
    """
    root = Path(path)
    if not root.is_dir():
        raise NotADirectoryError(f"Expected an artifact directory: {root}")
    digest = hashlib.sha256()
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Artifact directory is empty: {root}")
    for item in files:
        if item.is_symlink():
            raise ValueError(f"Refusing to fingerprint symlinked artifact file: {item}")
        rel = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(rel).to_bytes(8, "big"))
        digest.update(rel)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle)
    return content or {}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def hamming_hex(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def write_table(
    records: list[dict[str, Any]], csv_path: str | Path, parquet_path: str | Path | None = None
) -> str:
    """Atomically publish CSV and, when optional parquet support is installed, Parquet outputs."""
    import pandas as pd

    frame = pd.DataFrame(records)
    csv_path = Path(csv_path)
    ensure_dir(csv_path.parent)
    with tempfile.NamedTemporaryFile(
        prefix=f".{csv_path.stem}.", suffix=".csv", dir=csv_path.parent, delete=False
    ) as handle:
        csv_temp = Path(handle.name)
    try:
        frame.to_csv(csv_temp, index=False)
        os.replace(csv_temp, csv_path)
    finally:
        csv_temp.unlink(missing_ok=True)
    if parquet_path is not None:
        parquet_path = Path(parquet_path)
        ensure_dir(parquet_path.parent)
        with tempfile.NamedTemporaryFile(
            prefix=f".{parquet_path.stem}.",
            suffix=".parquet",
            dir=parquet_path.parent,
            delete=False,
        ) as handle:
            parquet_temp = Path(handle.name)
        try:
            frame.to_parquet(parquet_temp, index=False)
            os.replace(parquet_temp, parquet_path)
            return "parquet"
        except (ImportError, ValueError, ModuleNotFoundError):
            parquet_temp.unlink(missing_ok=True)
            return "csv_only"
        finally:
            parquet_temp.unlink(missing_ok=True)
    return "csv_only"
