from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, cast

from visualact_r.schemas import TaskRecord
from visualact_r.utils import stable_hash, write_json, write_jsonl


@dataclass
class _BKNode:
    value: int
    record_idx: int
    children: dict[int, "_BKNode"] = field(default_factory=dict)


def _hamming_int(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _bk_insert(root: _BKNode | None, value: int, record_idx: int) -> _BKNode:
    if root is None:
        return _BKNode(value=value, record_idx=record_idx)
    node = root
    while True:
        distance = _hamming_int(value, node.value)
        child = node.children.get(distance)
        if child is None:
            node.children[distance] = _BKNode(value=value, record_idx=record_idx)
            return root
        node = child


def _bk_query(root: _BKNode | None, value: int, radius: int) -> list[int]:
    if root is None:
        return []
    found: list[int] = []
    stack = [root]
    while stack:
        node = stack.pop()
        distance = _hamming_int(value, node.value)
        if distance <= radius:
            found.append(node.record_idx)
        lower, upper = distance - radius, distance + radius
        stack.extend(child for edge, child in node.children.items() if lower <= edge <= upper)
    return found


SPLITS = ["train", "dev", "calibration", "final"]


def assign_phash_clusters(records: list[TaskRecord], threshold: int) -> list[TaskRecord]:
    """Build deterministic complete-linkage pHash clusters with a BK-tree neighbor index.

    Complete linkage prevents the transitive-chain failure of connected-component clustering. The BK-tree avoids
    an all-pairs scan for large BDD-derived pools while preserving exact radius-neighbor discovery under Hamming
    distance. Each final cluster remains pHash-atomic for the split contract.
    """
    ordered = sorted(records, key=lambda item: item.image_id)
    hashes = [int(item.phash, 16) for item in ordered]
    clusters: list[list[int]] = []
    record_cluster: dict[int, int] = {}
    root: _BKNode | None = None
    for idx, value in enumerate(hashes):
        neighbor_indices = _bk_query(root, value, threshold)
        candidate_clusters = sorted({record_cluster[neighbor] for neighbor in neighbor_indices})
        compatible: list[tuple[float, int]] = []
        for cluster_idx in candidate_clusters:
            members = clusters[cluster_idx]
            distances = [_hamming_int(value, hashes[member]) for member in members]
            if distances and max(distances) <= threshold:
                compatible.append((sum(distances) / len(distances), cluster_idx))
        if compatible:
            _, cluster_idx = min(compatible, key=lambda item: (item[0], item[1]))
        else:
            cluster_idx = len(clusters)
            clusters.append([])
        clusters[cluster_idx].append(idx)
        record_cluster[idx] = cluster_idx
        root = _bk_insert(root, value, idx)
    canonical: dict[str, str] = {}
    for members in clusters:
        member_ids = sorted(ordered[idx].image_id for idx in members)
        cluster_id = f"phash_{stable_hash(member_ids)[:12]}"
        for idx in members:
            canonical[ordered[idx].image_id] = cluster_id
    return [replace(item, phash_cluster_id=canonical[item.image_id]) for item in records]


def _is_hard(record: TaskRecord) -> bool:
    return (
        bool(
            set(record.hard_flags)
            & {"occluded", "truncated", "small_object", "multiple_critical", "primary_zone"}
        )
        or record.timeofday in {"night", "dawn/dusk"}
        or record.weather in {"rainy", "foggy"}
    )


def _target_counts(total: int, config: dict[str, Any]) -> dict[str, int]:
    data_cfg = config["data"]
    targets = {
        "train": int(round(total * float(data_cfg["train_fraction"]))),
        "dev": int(round(total * float(data_cfg["dev_fraction"]))),
        "calibration": int(round(total * float(data_cfg["calibration_fraction"]))),
    }
    targets["final"] = total - sum(targets.values())
    return targets


def _desired_label_counts(
    global_counts: Counter[str], targets: dict[str, int], total: int
) -> dict[str, Counter[str]]:
    desired: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    for label, count in global_counts.items():
        allocated = 0
        for split in SPLITS[:-1]:
            value = int(round(count * targets[split] / max(1, total)))
            desired[split][label] = value
            allocated += value
        desired["final"][label] = count - allocated
    return desired


def _split_objective(
    split: str,
    members: list[TaskRecord],
    targets: dict[str, int],
    desired_actions: dict[str, Counter[str]],
    desired_hard: dict[str, float],
    counts: Counter[str],
    action_counts: dict[str, Counter[str]],
    hard_counts: Counter[str],
) -> tuple[float, str]:
    size = len(members)
    after_count = counts[split] + size
    capacity = ((after_count - targets[split]) / max(1, targets[split])) ** 2
    group_actions: Counter[str] = Counter(str(item.action_label) for item in members)
    action_cost = 0.0
    for label in desired_actions[split] | group_actions:
        after = action_counts[split][label] + group_actions[label]
        target = desired_actions[split][label]
        action_cost += ((after - target) / max(1.0, float(target))) ** 2
    group_hard = sum(_is_hard(item) for item in members)
    hard_cost = (
        (hard_counts[split] + group_hard - desired_hard[split]) / max(1.0, desired_hard[split])
    ) ** 2
    # Small stable tie break avoids order sensitivity without leaking labels across splits.
    return (4.0 * capacity + 2.0 * action_cost + 0.75 * hard_cost, split)


def _coverage_requirements(
    records: list[TaskRecord], targets: dict[str, int], config: dict[str, Any]
) -> dict[str, Any]:
    global_actions: Counter[str] = Counter(str(item.action_label) for item in records)
    # Only require every common action in every split when the smallest split can physically contain it.
    # Small fixtures instead require action diversity, preventing a degenerate one-class final set.
    can_cover_all_actions = (
        min(targets.values()) >= len(global_actions) and min(targets.values()) > 0
    )
    required_actions = {
        action
        for action, count in global_actions.items()
        if can_cover_all_actions and count >= len(SPLITS)
    }
    min_action_diversity = min(len(global_actions), min(targets.values())) if targets else 0
    hard_total = sum(_is_hard(item) for item in records)
    clean_total = len(records) - hard_total
    expected_final_hard = int(round(hard_total * targets["final"] / max(1, len(records))))
    expected_final_clean = int(round(clean_total * targets["final"] / max(1, len(records))))
    config_hard_minimum = int(config["data"].get("final_hard_minimum", 0))
    config_clean_minimum = int(config["data"].get("final_clean_minimum", 0))
    final_hard_required = min(config_hard_minimum, max(1, expected_final_hard)) if hard_total else 0
    final_clean_required = (
        min(config_clean_minimum, max(1, expected_final_clean)) if clean_total else 0
    )
    # A small final split cannot satisfy independently rounded hard and clean minimums if they exceed capacity.
    # Preserve both regimes when possible, then allocate remaining capacity to the larger requested requirement.
    final_capacity = targets["final"]
    if hard_total and clean_total and final_capacity >= 2:
        final_hard_required = min(final_hard_required, max(1, final_capacity - 1))
        final_clean_required = min(
            final_clean_required, max(1, final_capacity - final_hard_required)
        )
        if final_hard_required + final_clean_required > final_capacity:
            final_hard_required = max(1, min(final_hard_required, final_capacity - 1))
            final_clean_required = final_capacity - final_hard_required
    elif final_hard_required + final_clean_required > final_capacity:
        # This only happens for pathological tiny pools where both regimes cannot be represented.
        final_clean_required = max(0, final_capacity - final_hard_required)
    return {
        "required_action_coverage": sorted(required_actions),
        "min_action_diversity": min_action_diversity,
        "final_hard_required": final_hard_required,
        "final_clean_required": final_clean_required,
    }


def validate_split_integrity(
    records: list[TaskRecord], config: dict[str, Any] | None = None
) -> dict[str, Any]:
    seen_images: dict[str, str] = {}
    seen_clusters: dict[str, str] = {}
    image_overlap = 0
    cluster_overlap = 0
    for record in records:
        if record.image_id in seen_images and seen_images[record.image_id] != record.split:
            image_overlap += 1
        if (
            record.phash_cluster_id in seen_clusters
            and seen_clusters[record.phash_cluster_id] != record.split
        ):
            cluster_overlap += 1
        seen_images[record.image_id] = record.split
        seen_clusters[record.phash_cluster_id] = record.split
    targets = _target_counts(len(records), config) if config else {split: 1 for split in SPLITS}
    requirements = (
        _coverage_requirements(records, targets, config)
        if config
        else {
            "required_action_coverage": [],
            "min_action_diversity": 0,
            "final_hard_required": 0,
            "final_clean_required": 0,
        }
    )
    action_distribution: dict[str, dict[str, int]] = {
        split: dict(
            Counter(str(record.action_label) for record in records if record.split == split)
        )
        for split in SPLITS
    }
    required_actions = [
        str(action) for action in cast(list[str], requirements["required_action_coverage"])
    ]
    missing_action_coverage: dict[str, list[str]] = {
        split: [
            action for action in required_actions if action_distribution[split].get(action, 0) == 0
        ]
        for split in SPLITS
    }
    action_diversity = {split: len(action_distribution[split]) for split in SPLITS}
    typed_requirements = cast(dict[str, Any], requirements)
    minimum_action_diversity = int(typed_requirements["min_action_diversity"])
    diversity_pass = all(value >= minimum_action_diversity for value in action_diversity.values())
    final_hard_count = sum(record.split == "final" and _is_hard(record) for record in records)
    final_clean_count = sum(record.split == "final" and not _is_hard(record) for record in records)
    final_hard_required = int(typed_requirements["final_hard_required"])
    final_clean_required = int(typed_requirements["final_clean_required"])
    composition_pass = (
        all(not values for values in missing_action_coverage.values())
        and diversity_pass
        and final_hard_count >= final_hard_required
        and final_clean_count >= final_clean_required
    )
    return {
        "image_overlap_count": image_overlap,
        "phash_cluster_overlap_count": cluster_overlap,
        "action_distribution": action_distribution,
        "missing_action_coverage": missing_action_coverage,
        "action_diversity": action_diversity,
        "minimum_action_diversity": minimum_action_diversity,
        "final_hard_count": final_hard_count,
        "final_hard_required": final_hard_required,
        "final_clean_count": final_clean_count,
        "final_clean_required": final_clean_required,
        "composition_status": "PASS" if composition_pass else "HOLD",
        "status": "PASS"
        if image_overlap == 0 and cluster_overlap == 0 and composition_pass
        else "HOLD",
    }


def build_splits(
    records: list[TaskRecord], config: dict[str, Any], output_dir: str
) -> list[TaskRecord]:
    clustered = assign_phash_clusters(records, int(config["data"]["phash_distance_threshold"]))
    groups: dict[str, list[TaskRecord]] = defaultdict(list)
    for record in clustered:
        groups[record.phash_cluster_id].append(record)
    total = len(clustered)
    targets = _target_counts(total, config)
    global_actions: Counter[str] = Counter(str(record.action_label) for record in clustered)
    desired_actions = _desired_label_counts(global_actions, targets, total)
    global_hard = sum(_is_hard(record) for record in clustered)
    desired_hard = {split: global_hard * targets[split] / max(1, total) for split in SPLITS}
    counts: Counter[str] = Counter()
    action_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    hard_counts: Counter[str] = Counter()
    assigned: dict[str, str] = {}
    requirements = _coverage_requirements(clustered, targets, config)

    def assign_group(cluster: str, members: list[TaskRecord], destination: str) -> None:
        assigned[cluster] = destination
        counts[destination] += len(members)
        action_counts[destination].update(str(record.action_label) for record in members)
        hard_counts[destination] += sum(_is_hard(record) for record in members)

    # Reserve pHash-atomic final groups for both hard and clean coverage before the greedy fill.
    # This avoids a tiny, yet leakage-safe, final set degenerating into a single severity regime.
    for require_hard in (True, False):
        required_key = "final_hard_required" if require_hard else "final_clean_required"
        required = int(cast(dict[str, Any], requirements)[required_key])
        while (
            hard_counts["final"] if require_hard else counts["final"] - hard_counts["final"]
        ) < required:
            candidates = []
            for cluster, members in groups.items():
                if cluster in assigned or counts["final"] + len(members) > targets["final"]:
                    continue
                desired_count = sum(_is_hard(member) == require_hard for member in members)
                if desired_count:
                    candidates.append(
                        (len(members), -desired_count, stable_hash(cluster), cluster, members)
                    )
            if not candidates:
                break
            _, _, _, cluster, members = min(candidates)
            assign_group(cluster, members, "final")

    # Rarer action groups are placed first. Every choice is still pHash-atomic.
    ordered = sorted(
        [(cluster, members) for cluster, members in groups.items() if cluster not in assigned],
        key=lambda item: (
            min(global_actions[record.action_label] for record in item[1]),
            -len(item[1]),
            stable_hash(item[0]),
        ),
    )
    for cluster, members in ordered:
        # Prefer a split that can absorb the full pHash-atomic group without exceeding its target.
        # This avoids sending every early group to train simply because train has the largest desired label counts.
        feasible_splits = [
            split for split in SPLITS if counts[split] + len(members) <= targets[split]
        ]
        candidate_splits = feasible_splits or SPLITS
        destination = min(
            _split_objective(
                split,
                members,
                targets,
                desired_actions,
                desired_hard,
                counts,
                action_counts,
                hard_counts,
            )
            for split in candidate_splits
        )[1]
        assign_group(cluster, members, destination)
    out = [replace(record, split=assigned[record.phash_cluster_id]) for record in clustered]
    integrity = validate_split_integrity(out, config)
    final_hard = [record for record in out if record.split == "final" and _is_hard(record)]
    final_clean = [record for record in out if record.split == "final" and not _is_hard(record)]
    manifest = {
        "split_counts": {name: sum(record.split == name for record in out) for name in SPLITS},
        "final_hard_count": len(final_hard),
        "final_clean_count": len(final_clean),
        "cluster_count": len(groups),
        "algorithm": "pHash-atomic deterministic stratified group assignment",
        "integrity": integrity,
        "config_hash": stable_hash(config),
    }
    from pathlib import Path

    root = Path(output_dir)
    write_jsonl(root / "task_records.jsonl", [x.to_dict() for x in out])
    write_jsonl(root / "inference_records.jsonl", [x.inference_contract() for x in out])
    manifest["inference_contract_path"] = str(root / "inference_records.jsonl")
    manifest["inference_contract_excludes"] = [
        "action_label",
        "gt_objects",
        "supporting_object_ids",
        "hard_flags",
        "phash",
        "phash_cluster_id",
    ]
    write_json(root / "split_manifest.json", manifest)
    write_jsonl(
        root / "frozen_hard_ids.jsonl",
        [{"image_id": x.image_id, "task_id": x.task_id} for x in final_hard],
    )
    write_jsonl(
        root / "frozen_clean_ids.jsonl",
        [{"image_id": x.image_id, "task_id": x.task_id} for x in final_clean],
    )
    if integrity["status"] != "PASS":
        raise RuntimeError(f"Leakage-safe split composition validation failed: {integrity}")
    return out
