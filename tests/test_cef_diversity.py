from visualact_r.cef.core import _diverse_top_k
from visualact_r.schemas import Box, CropRecord


def _crop(crop_id: str, box: Box) -> CropRecord:
    return CropRecord(
        crop_id=crop_id,
        image_id="image",
        query_id="q",
        box_xyxy=box,
        primary_candidate_id=crop_id,
        raw_candidate_ids=[crop_id],
        category="car",
        base_confidence=0.9,
        edge_expansion=[0.0] * 4,
        edge_deflation=[0.0] * 4,
        coverage_proxy=0.9,
        contamination_proxy=0.1,
        selection_rank=0,
        variant="cef",
    )


def test_diverse_top_k_avoids_redundant_high_scoring_crop():
    left = _crop("left", Box(0, 0, 10, 10))
    overlap = _crop("overlap", Box(0, 0, 10, 10))
    distinct = _crop("distinct", Box(20, 0, 30, 10))
    selected = _diverse_top_k(
        [(0.99, left), (0.98, overlap), (0.60, distinct)], top_k=2, penalty=0.9
    )
    assert [crop.crop_id for crop in selected] == ["left", "distinct"]
