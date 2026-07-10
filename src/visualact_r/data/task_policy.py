from __future__ import annotations

from typing import Any, Iterable

from visualact_r.schemas import Action, Box, GroundTruthObject


def point_in_polygon(x: float, y: float, polygon: list[list[float]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        crossing = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        if crossing:
            inside = not inside
        j = i
    return inside


def normalized_center(box: Box, width: int, height: int) -> tuple[float, float]:
    x, y = box.center
    return x / width, y / height


def area_fraction(box: Box, width: int, height: int) -> float:
    return box.area / max(1.0, width * height)


def triage_policy(
    objects: Iterable[GroundTruthObject],
    width: int,
    height: int,
    config: dict[str, Any],
) -> tuple[Action, list[str], list[str]]:
    policy = config["task_policy"]
    critical = set(config["data"]["critical_categories"])
    primary = policy["primary_zone"]
    secondary = policy["secondary_zone"]
    large_area = float(policy["primary_large_area_fraction"])
    small_area = float(policy["small_area_fraction"])
    ambiguity_count = int(policy["ambiguity_count_threshold"])

    primary_hits: list[GroundTruthObject] = []
    secondary_hits: list[GroundTruthObject] = []
    ambiguous: list[GroundTruthObject] = []
    flags: list[str] = []
    for obj in objects:
        if obj.category not in critical:
            continue
        x, y = normalized_center(obj.box_xyxy, width, height)
        area = area_fraction(obj.box_xyxy, width, height)
        in_primary = point_in_polygon(x, y, primary)
        in_secondary = point_in_polygon(x, y, secondary)
        if in_primary:
            primary_hits.append(obj)
        if in_secondary and not in_primary:
            secondary_hits.append(obj)
        if obj.occluded:
            flags.append("occluded")
        if obj.truncated:
            flags.append("truncated")
        if area <= small_area:
            flags.append("small_object")
        if obj.occluded or obj.truncated or area <= small_area:
            ambiguous.append(obj)

    if (
        any(area_fraction(x.box_xyxy, width, height) >= large_area for x in primary_hits)
        or len(ambiguous) >= 2
    ):
        action: Action = "ESCALATE"
        supporting = [x.object_id for x in primary_hits] or [x.object_id for x in ambiguous]
    elif (
        primary_hits
        or secondary_hits
        or ambiguous
        or len(primary_hits) + len(secondary_hits) >= ambiguity_count
    ):
        action = "REVIEW"
        supporting = [x.object_id for x in primary_hits + secondary_hits + ambiguous]
    else:
        action = "AUTO_CLEAR"
        supporting = []

    if len(primary_hits) + len(secondary_hits) >= ambiguity_count:
        flags.append("multiple_critical")
    if primary_hits:
        flags.append("primary_zone")
    elif secondary_hits:
        flags.append("secondary_zone")
    return action, sorted(set(supporting)), sorted(set(flags))


def task_prompt(template_id: str = "triage_standard") -> str:
    if template_id == "triage_conservative":
        return (
            "Assess whether the generated urban-scene assessment can be auto-cleared, requires review, "
            "or should be escalated. Use only visible evidence and return a typed decision."
        )
    return (
        "Assess whether the generated urban-scene assessment can be AUTO_CLEARed, requires REVIEW, "
        "or should be ESCALATEd. Use only visible evidence and return a typed decision."
    )
