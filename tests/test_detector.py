from visualact_r.detector.synthetic import SyntheticColorDetector


def test_detector_exposes_raw_and_nms(records):
    raw, nms = SyntheticColorDetector().predict(records[0])
    assert raw
    assert nms
    assert len(raw) >= len(nms)
    assert all(x.pre_nms for x in raw)
    assert all(not x.pre_nms for x in nms)
    assert all(x.nms_cluster_id for x in raw)

from visualact_r.detector.ultralytics_adapter import UltralyticsRawCandidateAdapter
from visualact_r.schemas import Box, DetectionCandidate


def test_ultralytics_adapter_post_nms_fallback_is_explicit(records):
    adapter = UltralyticsRawCandidateAdapter(
        weights="dummy.pt", allow_post_nms_fallback=True, confidence=0.01
    )
    adapter._load = lambda: object()  # type: ignore[method-assign]
    adapter._raw_pre_nms_candidates = lambda model, record: []  # type: ignore[method-assign]

    def fallback(model, record):
        raw = DetectionCandidate(
            raw_candidate_id=f"{record.image_id}:fallback:0",
            image_id=record.image_id,
            category="car",
            box_xyxy=Box(1, 2, 10, 20),
            confidence=0.7,
            pre_nms=True,
            nms_cluster_id=f"{record.image_id}:fallback_cluster:0",
            source="ultralytics_post_nms_fallback_as_raw:dummy.pt",
        )
        nms = DetectionCandidate(
            raw_candidate_id=raw.raw_candidate_id,
            image_id=record.image_id,
            category="car",
            box_xyxy=raw.box_xyxy,
            confidence=raw.confidence,
            pre_nms=False,
            nms_cluster_id=raw.nms_cluster_id,
            source="ultralytics_post_nms_fallback:dummy.pt",
        )
        return [raw], [nms]

    adapter._post_nms_fallback_candidates = fallback  # type: ignore[method-assign]
    raw, nms = adapter.predict(records[0])

    assert adapter.last_used_post_nms_fallback is True
    assert raw and nms
    assert raw[0].pre_nms is True
    assert nms[0].pre_nms is False
    assert "post_nms_fallback" in raw[0].source
