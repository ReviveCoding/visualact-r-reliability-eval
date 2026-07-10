from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from visualact_r.utils import ensure_dir, write_json

CATEGORY_COLORS: dict[str, tuple[int, int, int]] = {
    "person": (220, 30, 30),
    "rider": (230, 120, 30),
    "bike": (230, 210, 30),
    "motor": (140, 40, 190),
    "car": (35, 120, 230),
    "truck": (30, 180, 90),
    "bus": (40, 190, 190),
}


def _object(
    category: str,
    box: tuple[int, int, int, int],
    idx: int,
    occluded: bool = False,
    truncated: bool = False,
) -> dict[str, Any]:
    return {
        "id": idx,
        "category": category,
        "attributes": {"occluded": occluded, "truncated": truncated, "trafficLightColor": "none"},
        "box2d": {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]},
    }


def create_synthetic_fixture(root: str | Path, count: int = 24) -> dict[str, Any]:
    root = Path(root)
    images = ensure_dir(root / "images")
    labels: list[dict[str, Any]] = []
    width, height = 640, 360
    for i in range(count):
        image_id = f"fixture_{i:03d}.png"
        base = Image.new(
            "RGB", (width, height), (50 + (i % 3) * 15, 55 + (i % 4) * 10, 60 + (i % 5) * 8)
        )
        draw = ImageDraw.Draw(base)
        # Road wedge and horizon provide image content without using GT at detector inference.
        draw.polygon([(260, 170), (380, 170), (610, 359), (30, 359)], fill=(70, 70, 70))
        draw.rectangle((0, 160, width, 170), fill=(100, 100, 100))
        # Deterministic visual marker keeps non-duplicate fixture scenes distinct under pHash.
        # fixture_001 is overwritten from fixture_000 below to retain one intentional duplicate pair.
        marker_col = 16 + (i * 37) % 180
        marker_x = 12 + (i % 6) * 96
        marker_y = 14 + ((i // 6) % 4) * 30
        draw.rectangle((marker_x, marker_y, marker_x + 52, marker_y + 18), fill=(marker_col,) * 3)
        objs: list[dict[str, Any]] = []
        if i % 3 == 0:
            # Large primary-zone car -> ESCALATE.
            candidates = [("car", (275, 210, 390, 340), False, False)]
            if i % 6 == 0:
                candidates.append(("person", (245, 220, 270, 305), True, False))
        elif i % 3 == 1:
            # Small / secondary / occluded object -> REVIEW.
            candidates = [("person", (185, 225, 210, 300), True, False)]
            if i % 4 == 1:
                candidates.append(("bike", (410, 250, 455, 310), False, False))
        else:
            # Context object deliberately outside review zones -> AUTO_CLEAR.
            candidates = [("car", (8, 174, 68, 220), False, False)]
            if i % 5 == 2:
                candidates.append(("bus", (520, 172, 625, 220), False, False))
        for j, (category, box, occluded, truncated) in enumerate(candidates):
            color = CATEGORY_COLORS[category]
            draw.rectangle(box, fill=color, outline=(255, 255, 255), width=1)
            objs.append(_object(category, box, j, occluded, truncated))
        # Near duplicate pair, retained for pHash cluster testing.
        if i == 1:
            src = images / "fixture_000.png"
            if src.exists():
                base = Image.open(src).copy()
        base.save(images / image_id)
        labels.append(
            {
                "name": image_id,
                "attributes": {
                    # Metadata severity is intentionally decoupled from the action pattern so fixture final splits
                    # contain both clean and hard slices. This tests split/evaluation mechanics, not scene realism.
                    "weather": ["clear", "overcast", "clear", "clear", "rainy", "foggy"][i % 6],
                    "scene": ["city street", "highway", "residential"][i % 3],
                    "timeofday": ["daytime", "daytime", "daytime", "daytime", "night", "dawn/dusk"][
                        i % 6
                    ],
                },
                "timestamp": 10000 + i,
                "labels": objs,
            }
        )
    annotation_path = root / "fixture_annotations.json"
    write_json(annotation_path, labels)
    malformed = [
        {"name": "missing_labels.png", "attributes": {}},
        {"name": "bad_box.png", "labels": [{"category": "car", "box2d": {"x1": 5}}]},
        {
            "name": "bad_category.png",
            "labels": [{"category": "unknown", "box2d": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}}],
        },
    ]
    write_json(root / "malformed_examples.json", malformed)
    return {
        "root": str(root),
        "images_dir": str(images),
        "annotations": str(annotation_path),
        "count": count,
    }
