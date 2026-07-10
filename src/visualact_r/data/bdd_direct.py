"""Direct adapter for an already-extracted BDD100K train/validation layout.

The adapter deliberately does not call the archive-mode ``preflight`` or ``extract-bdd``
commands. It accepts the local layout already validated by Phase A::

    <bdd-root>/Images/train
    <bdd-root>/Images/val
    <bdd-root>/Labels/train
    <bdd-root>/Labels/val

BDD ``train`` is the only source for train/dev/calibration. BDD ``val`` is copied into
an immutable ``final`` partition at first build and is never used by the development
splitter. The resulting inference artifact remains label-free by construction.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Any, Iterable, Literal

from PIL import Image

from visualact_r.artifacts import save_inference_records, save_records
from visualact_r.data.build import _annotation_objects, perceptual_hash_image
from visualact_r.data.splits import assign_phash_clusters
from visualact_r.data.task_policy import task_prompt, triage_policy
from visualact_r.schemas import InferenceRecord, TaskRecord
from visualact_r.utils import ensure_dir, sha256_file, stable_hash, write_json, write_jsonl

_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
_JSON_SUFFIXES = frozenset({".json", ".jsonl"})
_DEVELOPMENT_SPLITS = ("train", "dev", "calibration")


class BDDDirectLayoutError(ValueError):
    """Raised when an extracted BDD layout cannot support the strict source contract."""


def _required_paths(bdd_root: Path) -> dict[str, Path]:
    return {
        "images_train": bdd_root / "Images" / "train",
        "images_val": bdd_root / "Images" / "val",
        "labels_train": bdd_root / "Labels" / "train",
        "labels_val": bdd_root / "Labels" / "val",
    }


def _image_index(images_dir: Path) -> dict[str, Path]:
    images = sorted(
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )
    output: dict[str, Path] = {}
    duplicates: list[str] = []
    for image in images:
        key = image.name
        if key in output:
            duplicates.append(key)
        else:
            output[key] = image
    if duplicates:
        raise BDDDirectLayoutError(
            f"Duplicate image basenames in {images_dir}: {sorted(set(duplicates))[:5]}"
        )
    return output


def _json_files(labels_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in labels_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in _JSON_SUFFIXES
    )


def _normalize_rows(payload: Any, source: Path) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        if isinstance(payload.get("frames"), list):
            rows = payload["frames"]
        elif isinstance(payload.get("data"), list):
            rows = payload["data"]
        else:
            rows = [payload]
    else:
        raise BDDDirectLayoutError(
            f"Unsupported annotation JSON type in {source}: {type(payload).__name__}"
        )
    if not all(isinstance(row, dict) for row in rows):
        raise BDDDirectLayoutError(f"Annotation rows must be JSON objects in {source}")
    return [dict(row) for row in rows]


_ROW_NAME_KEYS = (
    "name",
    "image_name",
    "imageName",
    "filename",
    "file_name",
    "image_path",
    "imagePath",
)
_NESTED_ROW_NAME_KEYS = ("image", "frame", "metadata", "meta")


def _row_image_name(row: dict[str, Any]) -> str | None:
    """Read a filename from common frame-level schema variants without guessing paths."""
    for key in _ROW_NAME_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return Path(value).name
    for container_key in _NESTED_ROW_NAME_KEYS:
        nested = row.get(container_key)
        if not isinstance(nested, dict):
            continue
        for key in _ROW_NAME_KEYS:
            value = nested.get(key)
            if isinstance(value, str) and value.strip():
                return Path(value).name
    return None


def _infer_name_from_annotation_file(path: Path, image_index: dict[str, Path]) -> str | None:
    """Infer an image name only when the annotation filename maps to one image exactly.

    Official BDD frame rows carry ``name``. Some already-extracted, one-file-per-image
    layouts instead use ``<image-stem>.json`` and omit that field. For those layouts,
    resolve the filename stem against the actual image index, never by assuming a suffix.
    """
    stem_or_name = path.name
    if stem_or_name.lower().endswith(".jsonl"):
        stem_or_name = stem_or_name[:-6]
    elif stem_or_name.lower().endswith(".json"):
        stem_or_name = stem_or_name[:-5]
    direct_name = Path(stem_or_name).name
    if direct_name in image_index:
        return direct_name

    stem = Path(direct_name).stem
    candidates = sorted(image_name for image_name in image_index if Path(image_name).stem == stem)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise BDDDirectLayoutError(
            "Annotation filename stem maps to multiple images and cannot be inferred safely: "
            f"{path} -> {candidates[:5]}"
        )
    return None


def _load_annotation_rows(labels_dir: Path, image_index: dict[str, Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for annotation_file in _json_files(labels_dir):
        if annotation_file.suffix.lower() == ".jsonl":
            with annotation_file.open("r", encoding="utf-8") as handle:
                payload_rows = [json.loads(line) for line in handle if line.strip()]
        else:
            with annotation_file.open("r", encoding="utf-8") as handle:
                payload_rows = _normalize_rows(json.load(handle), annotation_file)
        inferred_name = _infer_name_from_annotation_file(annotation_file, image_index)
        for row in payload_rows:
            row_name = _row_image_name(row)
            if row_name is not None:
                row["name"] = row_name
            elif inferred_name is not None:
                row["name"] = inferred_name
            else:
                raise BDDDirectLayoutError(
                    "Annotation row has no image name and its filename cannot be matched to an "
                    f"image uniquely: {annotation_file}"
                )
            rows.append(row)
    if not rows:
        raise BDDDirectLayoutError(f"No usable annotation rows found below: {labels_dir}")
    return rows


def _validate_rows_against_images(
    rows: Iterable[dict[str, Any]], image_index: dict[str, Path], source_name: str
) -> dict[str, Any]:
    counts: Counter[str] = Counter(str(row["name"]) for row in rows)
    duplicate_names = sorted(name for name, count in counts.items() if count != 1)
    missing_images = sorted(set(counts) - set(image_index))
    return {
        "source": source_name,
        "annotation_row_count": sum(counts.values()),
        "unique_annotation_image_count": len(counts),
        "image_count": len(image_index),
        "duplicate_annotation_names": duplicate_names[:20],
        "duplicate_annotation_name_count": len(duplicate_names),
        "missing_image_names": missing_images[:20],
        "missing_image_count": len(missing_images),
        "unlabeled_image_count": len(set(image_index) - set(counts)),
        "status": "PASS" if not duplicate_names and not missing_images else "HOLD",
    }


def inspect_extracted_bdd_layout(bdd_root: str | Path) -> dict[str, Any]:
    """Read only metadata and annotations, validating exact image-label joins.

    This does not open images and does not write task records. It is the intended first
    command before a potentially long pHash/task-record build.
    """
    root = Path(bdd_root).expanduser().resolve()
    paths = _required_paths(root)
    missing = sorted(name for name, path in paths.items() if not path.is_dir())
    if missing:
        raise BDDDirectLayoutError(
            f"Extracted BDD layout is incomplete below {root}; missing directories: {missing}"
        )
    train_images = _image_index(paths["images_train"])
    val_images = _image_index(paths["images_val"])
    train_rows = _load_annotation_rows(paths["labels_train"], train_images)
    val_rows = _load_annotation_rows(paths["labels_val"], val_images)
    train_report = _validate_rows_against_images(train_rows, train_images, "bdd_train")
    val_report = _validate_rows_against_images(val_rows, val_images, "bdd_val")
    exact_overlap = sorted(set(train_images).intersection(val_images))
    status = (
        "PASS"
        if train_report["status"] == val_report["status"] == "PASS" and not exact_overlap
        else "HOLD"
    )
    return {
        "status": status,
        "bdd_root": str(root),
        "layout_contract": {
            "development_source": "Images/train + Labels/train only",
            "frozen_final_source": "Images/val + Labels/val only",
            "archive_mode_commands_used": False,
        },
        "train": train_report,
        "final": val_report,
        "cross_source_exact_image_name_overlap": exact_overlap[:20],
        "cross_source_exact_image_name_overlap_count": len(exact_overlap),
        "label_file_count": {
            "train": len(_json_files(paths["labels_train"])),
            "val": len(_json_files(paths["labels_val"])),
        },
    }


def _records_from_rows(
    rows: Iterable[dict[str, Any]], image_index: dict[str, Path], config: dict[str, Any]
) -> list[TaskRecord]:
    records: list[TaskRecord] = []
    for row in sorted(rows, key=lambda item: str(item["name"])):
        name = str(row["name"])
        image_path = image_index[name]
        with Image.open(image_path) as image:
            width, height = image.size
            phash = perceptual_hash_image(image)
        objects = _annotation_objects(row)
        action, supporting, flags = triage_policy(objects, width, height, config)
        attrs = row.get("attributes", {})
        records.append(
            TaskRecord(
                image_id=name,
                image_path=str(image_path),
                width=width,
                height=height,
                task_id=f"{Path(name).stem}:triage_standard",
                task_template_id="triage_standard",
                prompt=task_prompt("triage_standard"),
                weather=str(attrs.get("weather", "unknown")),
                scene=str(attrs.get("scene", "unknown")),
                timeofday=str(attrs.get("timeofday", "unknown")),
                gt_objects=objects,
                action_label=action,
                supporting_object_ids=supporting,
                hard_flags=flags,
                phash=phash,
                label_policy_version=config["project"]["task_policy_version"],
            )
        )
    return records


def _development_targets(total: int, config: dict[str, Any]) -> dict[str, int]:
    fractions = {name: float(config["data"][f"{name}_fraction"]) for name in _DEVELOPMENT_SPLITS}
    denominator = sum(fractions.values())
    if denominator <= 0:
        raise ValueError("train/dev/calibration fractions must sum to a positive value")
    targets = {name: int(round(total * value / denominator)) for name, value in fractions.items()}
    targets["train"] += total - sum(targets.values())
    return targets


def _is_hard(record: TaskRecord) -> bool:
    return (
        bool(
            set(record.hard_flags)
            & {"occluded", "truncated", "small_object", "multiple_critical", "primary_zone"}
        )
        or record.timeofday in {"night", "dawn/dusk"}
        or record.weather in {"rainy", "foggy"}
    )


def _build_development_splits(
    records: list[TaskRecord], config: dict[str, Any]
) -> list[TaskRecord]:
    """Assign pHash-atomic train/dev/calibration groups without ever creating ``final``.

    The final partition is reserved exclusively for the upstream BDD validation source.
    """
    clustered = assign_phash_clusters(records, int(config["data"]["phash_distance_threshold"]))
    groups: dict[str, list[TaskRecord]] = {}
    for record in clustered:
        groups.setdefault(record.phash_cluster_id, []).append(record)
    targets = _development_targets(len(clustered), config)
    global_actions = Counter(str(record.action_label) for record in clustered)
    global_hard = sum(_is_hard(record) for record in clustered)
    desired_actions = {
        split: {
            action: count * targets[split] / max(1, len(clustered))
            for action, count in global_actions.items()
        }
        for split in _DEVELOPMENT_SPLITS
    }
    desired_hard = {
        split: global_hard * targets[split] / max(1, len(clustered))
        for split in _DEVELOPMENT_SPLITS
    }
    counts: Counter[str] = Counter()
    action_counts: dict[str, Counter[str]] = {split: Counter() for split in _DEVELOPMENT_SPLITS}
    hard_counts: Counter[str] = Counter()
    assignments: dict[str, str] = {}

    def objective(split: str, members: list[TaskRecord]) -> tuple[float, str]:
        size = len(members)
        capacity = ((counts[split] + size - targets[split]) / max(1, targets[split])) ** 2
        actions = Counter(str(member.action_label) for member in members)
        action_cost = sum(
            (
                (action_counts[split][action] + value - desired_actions[split][action])
                / max(1.0, desired_actions[split][action])
            )
            ** 2
            for action, value in actions.items()
        )
        hard_cost = (
            (hard_counts[split] + sum(_is_hard(member) for member in members) - desired_hard[split])
            / max(1.0, desired_hard[split])
        ) ** 2
        return (4.0 * capacity + 2.0 * action_cost + 0.75 * hard_cost, split)

    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (
            min(global_actions[str(record.action_label)] for record in item[1]),
            -len(item[1]),
            stable_hash(item[0]),
        ),
    )
    for cluster_id, members in ordered_groups:
        feasible = [
            split for split in _DEVELOPMENT_SPLITS if counts[split] + len(members) <= targets[split]
        ]
        destination = min(
            objective(split, members) for split in (feasible or list(_DEVELOPMENT_SPLITS))
        )[1]
        assignments[cluster_id] = destination
        counts[destination] += len(members)
        action_counts[destination].update(str(member.action_label) for member in members)
        hard_counts[destination] += sum(_is_hard(member) for member in members)
    return [replace(record, split=assignments[record.phash_cluster_id]) for record in clustered]


def _direct_integrity(
    development_records: list[TaskRecord], final_records: list[TaskRecord]
) -> dict[str, Any]:
    by_image: dict[str, str] = {}
    by_cluster: dict[str, str] = {}
    image_overlap = 0
    cluster_overlap = 0
    for record in [*development_records, *final_records]:
        existing = by_image.get(record.image_id)
        if existing is not None and existing != record.split:
            image_overlap += 1
        by_image[record.image_id] = record.split
        existing_cluster = by_cluster.get(record.phash_cluster_id)
        if existing_cluster is not None and existing_cluster != record.split:
            cluster_overlap += 1
        by_cluster[record.phash_cluster_id] = record.split
    development_splits = {record.split for record in development_records}
    final_splits = {record.split for record in final_records}
    return {
        "status": "PASS"
        if image_overlap == 0
        and cluster_overlap == 0
        and development_splits.issubset(set(_DEVELOPMENT_SPLITS))
        and final_splits == {"final"}
        else "HOLD",
        "image_overlap_count": image_overlap,
        "phash_cluster_overlap_count": cluster_overlap,
        "development_splits": sorted(development_splits),
        "final_splits": sorted(final_splits),
        "development_record_count": len(development_records),
        "frozen_final_record_count": len(final_records),
    }


def _phash_to_bits(phash: str) -> list[bool]:
    value = int(phash, 16)
    width = max(64, len(phash) * 4)
    return [bool((value >> shift) & 1) for shift in range(width - 1, -1, -1)]


def _cross_partition_phash_audit_cpu_bktree(
    development_records: list[TaskRecord], final_records: list[TaskRecord], threshold: int
) -> tuple[int, list[dict[str, Any]]]:
    """CPU Hamming-radius search using the same BK-tree primitive as split clustering."""
    from visualact_r.data.splits import _BKNode, _bk_insert, _bk_query

    root: _BKNode | None = None
    final_hashes = [int(record.phash, 16) for record in final_records]
    for index, value in enumerate(final_hashes):
        root = _bk_insert(root, value, index)
    examples: list[dict[str, Any]] = []
    match_count = 0
    for record in development_records:
        value = int(record.phash, 16)
        for index in _bk_query(root, value, threshold):
            match_count += 1
            if len(examples) < 20:
                final = final_records[index]
                examples.append(
                    {
                        "development_image_id": record.image_id,
                        "final_image_id": final.image_id,
                        "hamming_distance": (value ^ int(final.phash, 16)).bit_count(),
                    }
                )
    return match_count, examples


def _cross_partition_phash_audit_cuda(
    development_records: list[TaskRecord],
    final_records: list[TaskRecord],
    threshold: int,
    chunk_size: int,
) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    """Vectorized CUDA Hamming scan for fixed-size perceptual hashes.

    The pHash values themselves are already deterministic CPU artifacts. CUDA is used only
    for the many pairwise XOR + population-count-equivalent comparisons.
    """
    if chunk_size <= 0:
        raise ValueError("phash_chunk_size must be positive")
    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised only when optional GPU deps are absent
        raise RuntimeError("PyTorch is required for CUDA pHash comparison") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA pHash comparison requested but torch.cuda.is_available() is false")

    device = torch.device("cuda")
    start_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    final_bits = torch.tensor(
        [_phash_to_bits(record.phash) for record in final_records],
        dtype=torch.bool,
        device=device,
    )
    match_count = 0
    examples: list[dict[str, Any]] = []
    for start_index in range(0, len(development_records), chunk_size):
        chunk = development_records[start_index : start_index + chunk_size]
        dev_bits = torch.tensor(
            [_phash_to_bits(record.phash) for record in chunk],
            dtype=torch.bool,
            device=device,
        )
        # [chunk, final, bit] -> [chunk, final] Hamming distances.
        distances = torch.bitwise_xor(dev_bits[:, None, :], final_bits[None, :, :]).sum(dim=2)
        matches = distances <= threshold
        match_count += int(matches.sum().item())
        if len(examples) < 20 and bool(matches.any().item()):
            match_indices = matches.nonzero(as_tuple=False).detach().cpu().tolist()
            for local_dev_index, final_index in match_indices:
                if len(examples) >= 20:
                    break
                dev = chunk[local_dev_index]
                final = final_records[final_index]
                examples.append(
                    {
                        "development_image_id": dev.image_id,
                        "final_image_id": final.image_id,
                        "hamming_distance": int(distances[local_dev_index, final_index].item()),
                    }
                )
        del dev_bits, distances, matches
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time
    stats = {
        "device": str(torch.cuda.get_device_name(device)),
        "chunk_size": chunk_size,
        "elapsed_seconds": elapsed,
        "cuda_peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    return match_count, examples, stats


def _cross_partition_phash_audit(
    development_records: list[TaskRecord],
    final_records: list[TaskRecord],
    threshold: int,
    *,
    backend: Literal["auto", "cpu", "cuda"] = "auto",
    chunk_size: int = 256,
) -> dict[str, Any]:
    """Detect near-duplicate pHashes crossing the BDD train/val source boundary.

    Exact source split separation alone is weaker than a visual near-duplicate audit.
    The adapter does not silently drop BDD validation examples. It reports and, by default,
    blocks a build whenever a cross-boundary match is found.
    """
    if threshold < 0:
        raise ValueError("cross_partition_phash_threshold must be non-negative")
    if backend not in {"auto", "cpu", "cuda"}:
        raise ValueError("phash_compare_backend must be one of: auto, cpu, cuda")
    if chunk_size <= 0:
        raise ValueError("phash_chunk_size must be positive")

    started = time.perf_counter()
    backend_used = "cpu"
    backend_error: str | None = None
    cuda_stats: dict[str, Any] | None = None
    if backend in {"auto", "cuda"}:
        try:
            match_count, examples, cuda_stats = _cross_partition_phash_audit_cuda(
                development_records, final_records, threshold, chunk_size
            )
            backend_used = "cuda"
        except Exception as exc:
            backend_error = f"{type(exc).__name__}: {exc}"
            if backend == "cuda":
                raise
            match_count, examples = _cross_partition_phash_audit_cpu_bktree(
                development_records, final_records, threshold
            )
    else:
        match_count, examples = _cross_partition_phash_audit_cpu_bktree(
            development_records, final_records, threshold
        )
    elapsed = time.perf_counter() - started
    return {
        "status": "PASS" if match_count == 0 else "HOLD",
        "threshold": threshold,
        "match_count": match_count,
        "examples": examples,
        "compare_backend_requested": backend,
        "compare_backend_used": backend_used,
        "compare_backend_error": backend_error,
        "chunk_size": chunk_size,
        "elapsed_seconds": elapsed,
        "cuda": cuda_stats,
        "contract": "No pHash near-duplicate may cross BDD train-derived development and BDD val frozen-final by default.",
    }


def _frozen_final_records(records: list[TaskRecord]) -> list[TaskRecord]:
    """Assign unique final-only cluster ids so source validation cannot share a development cluster id."""
    return [
        replace(
            record,
            split="final",
            phash_cluster_id=f"bdd_val_frozen_{stable_hash(record.image_id)[:16]}",
        )
        for record in records
    ]


def _direct_build_manifest(
    *,
    status: str,
    root: Path,
    audit: dict[str, Any],
    max_train_records: int | None,
    integrity: dict[str, Any] | None,
    cross_partition_phash: dict[str, Any] | None,
    allow_cross_partition_phash: bool,
    config: dict[str, Any],
    error: str | None = None,
) -> dict[str, Any]:
    """Create a complete build manifest even when a leakage gate holds the run.

    A pHash scan can be expensive. Persisting the exact HOLD decision and examples prevents a
    long scan from ending with an opaque exception and no inspection artifact.
    """
    return {
        "status": status,
        "scope": "bdd100k_direct_extracted_layout",
        "protocol": {
            "development_source": "BDD100K train only",
            "development_splits": list(_DEVELOPMENT_SPLITS),
            "frozen_final_source": "BDD100K val only",
            "frozen_final_reassigned": False,
            "archive_mode_commands_used": False,
        },
        "bdd_root": str(root),
        "source_audit": audit,
        "development_sample": max_train_records is not None,
        "max_train_records": max_train_records,
        "integrity": integrity,
        "cross_partition_phash": cross_partition_phash,
        "cross_partition_phash_exception_allowed": allow_cross_partition_phash,
        "task_records_path": None,
        "task_records_sha256": None,
        "inference_records_path": None,
        "inference_records_sha256": None,
        "error": error,
        "claim_boundary": (
            "Development sample only: it validates local adapter behavior, not frozen-final performance."
            if max_train_records is not None
            else "Direct source-boundary build only: do not make performance claims until sealed final comparison."
        ),
        "config_hash": stable_hash(config),
    }


def build_direct_bdd_records(
    bdd_root: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
    *,
    max_train_records: int | None = None,
    cross_partition_phash_threshold: int = 4,
    allow_cross_partition_phash: bool = False,
    phash_compare_backend: Literal["auto", "cpu", "cuda"] = "cpu",
    phash_chunk_size: int = 256,
) -> dict[str, Any]:
    """Build direct BDD records while preserving BDD val as the sole frozen final source.

    ``max_train_records`` is an explicitly marked development sample. It never reduces
    BDD val, but it also cannot support final-performance claims. A HOLD manifest is written
    before returning when integrity or cross-partition pHash gates block publication.
    """
    audit = inspect_extracted_bdd_layout(bdd_root)
    if audit["status"] != "PASS":
        raise BDDDirectLayoutError(f"BDD direct adapter audit failed: {audit}")
    root = Path(bdd_root).expanduser().resolve()
    output = ensure_dir(output_dir)
    manifest_path = output / "direct_bdd_split_manifest.json"
    write_json(
        manifest_path,
        {
            "status": "RUNNING",
            "stage": "source_audit_pass",
            "bdd_root": str(root),
            "source_audit": audit,
            "claim_boundary": "Build in progress. No task records are claim-ready until final status is PASS.",
        },
    )
    paths = _required_paths(root)
    train_images, final_images = (
        _image_index(paths["images_train"]),
        _image_index(paths["images_val"]),
    )
    train_rows, final_rows = (
        _load_annotation_rows(paths["labels_train"], train_images),
        _load_annotation_rows(paths["labels_val"], final_images),
    )
    if max_train_records is not None:
        if max_train_records <= 0:
            raise ValueError("max_train_records must be positive when configured")
        train_rows = sorted(train_rows, key=lambda row: str(row["name"]))[:max_train_records]
    development_records = _build_development_splits(
        _records_from_rows(train_rows, train_images, config), config
    )
    write_json(
        manifest_path,
        {
            "status": "RUNNING",
            "stage": "development_records_built",
            "bdd_root": str(root),
            "development_record_count": len(development_records),
            "frozen_final_record_count": None,
            "claim_boundary": "Build in progress. Frozen final and cross-partition gates are not complete.",
        },
    )
    frozen_final_records = _frozen_final_records(
        _records_from_rows(final_rows, final_images, config)
    )
    write_json(
        manifest_path,
        {
            "status": "RUNNING",
            "stage": "frozen_final_records_built",
            "bdd_root": str(root),
            "development_record_count": len(development_records),
            "frozen_final_record_count": len(frozen_final_records),
            "phash_compare_backend_requested": phash_compare_backend,
            "phash_chunk_size": phash_chunk_size,
            "claim_boundary": "Build in progress. Cross-partition pHash gate is not complete.",
        },
    )
    integrity = _direct_integrity(development_records, frozen_final_records)
    cross_partition_phash = _cross_partition_phash_audit(
        development_records,
        frozen_final_records,
        cross_partition_phash_threshold,
        backend=phash_compare_backend,
        chunk_size=phash_chunk_size,
    )
    if integrity["status"] != "PASS":
        manifest = _direct_build_manifest(
            status="HOLD",
            root=root,
            audit=audit,
            max_train_records=max_train_records,
            integrity=integrity,
            cross_partition_phash=cross_partition_phash,
            allow_cross_partition_phash=allow_cross_partition_phash,
            config=config,
            error="Direct BDD source-boundary integrity gate blocked task-record publication.",
        )
        write_json(manifest_path, manifest)
        return manifest
    if cross_partition_phash["status"] != "PASS" and not allow_cross_partition_phash:
        manifest = _direct_build_manifest(
            status="HOLD",
            root=root,
            audit=audit,
            max_train_records=max_train_records,
            integrity=integrity,
            cross_partition_phash=cross_partition_phash,
            allow_cross_partition_phash=allow_cross_partition_phash,
            config=config,
            error=(
                "Cross-partition pHash leakage gate blocked task-record publication. "
                "Review cross_partition_phash.examples; do not bypass without documenting "
                "the exception and claim impact."
            ),
        )
        write_json(manifest_path, manifest)
        return manifest
    all_records = [*development_records, *frozen_final_records]
    save_records(output / "task_records.jsonl", all_records)
    save_inference_records(
        output / "inference_records.jsonl",
        [InferenceRecord.from_task_record(record) for record in all_records],
    )
    write_jsonl(
        output / "frozen_final_ids.jsonl",
        [
            {"image_id": record.image_id, "task_id": record.task_id}
            for record in frozen_final_records
        ],
    )
    manifest = _direct_build_manifest(
        status="PASS",
        root=root,
        audit=audit,
        max_train_records=max_train_records,
        integrity=integrity,
        cross_partition_phash=cross_partition_phash,
        allow_cross_partition_phash=allow_cross_partition_phash,
        config=config,
    )
    manifest.update(
        {
            "record_counts": {
                split: sum(record.split == split for record in all_records)
                for split in (*_DEVELOPMENT_SPLITS, "final")
            },
            "task_records_path": str(output / "task_records.jsonl"),
            "task_records_sha256": sha256_file(output / "task_records.jsonl"),
            "inference_records_path": str(output / "inference_records.jsonl"),
            "inference_records_sha256": sha256_file(output / "inference_records.jsonl"),
        }
    )
    write_json(manifest_path, manifest)
    return manifest
