from visualact_r.detector.ultralytics_adapter import RawCandidateExtractionUnsupported
from visualact_r.schemas import Box, DetectionCandidate
from visualact_r.utils import read_jsonl
from visualact_r.stages import build_data_stage, cache_detector_stage, train_cef_stage


def test_stage_commands_are_callable(tmp_path, fixture_source):
    build = build_data_stage(
        fixture_source["annotations"], fixture_source["images_dir"], str(tmp_path / "data")
    )
    assert build["status"] == "PASS"
    records = tmp_path / "data" / "task_records.jsonl"
    detector = cache_detector_stage(str(records), str(tmp_path / "detector"), "synthetic")
    assert detector["raw_candidate_extraction_validated"]
    cef = train_cef_stage(
        str(records),
        str(tmp_path / "detector" / "detector_raw_candidates.jsonl"),
        str(tmp_path / "detector" / "detector_nms_candidates.jsonl"),
        str(tmp_path / "cef"),
    )
    assert cef["status"] == "PASS"

from visualact_r.detector.ultralytics_adapter import UltralyticsRawCandidateAdapter


def test_cache_detector_writes_failure_report(tmp_path, fixture_source, monkeypatch):
    build = build_data_stage(
        fixture_source["annotations"], fixture_source["images_dir"], str(tmp_path / "data_fail")
    )
    assert build["status"] == "PASS"
    records = tmp_path / "data_fail" / "task_records.jsonl"

    def fail_predict(self, record):
        raise RuntimeError("forced detector failure")

    monkeypatch.setattr(UltralyticsRawCandidateAdapter, "predict", fail_predict)
    report = cache_detector_stage(
        str(records),
        str(tmp_path / "detector_fail"),
        "ultralytics",
        weights="dummy.pt",
    )

    assert report["status"] == "FAIL"
    assert report["detector_failures"]
    assert (tmp_path / "detector_fail" / "detector_cache_report.json").exists()


def test_cache_detector_exposes_ultralytics_runtime_options(tmp_path, fixture_source, monkeypatch):
    build = build_data_stage(
        fixture_source["annotations"], fixture_source["images_dir"], str(tmp_path / "data_opts")
    )
    assert build["status"] == "PASS"
    records = tmp_path / "data_opts" / "task_records.jsonl"

    def synthetic_predict(self, record):
        raw = DetectionCandidate(
            raw_candidate_id=f"{record.image_id}:raw:0",
            image_id=record.image_id,
            category="car",
            box_xyxy=Box(1, 2, 10, 20),
            confidence=0.7,
            pre_nms=True,
            nms_cluster_id=f"{record.image_id}:cluster:0",
            source="test",
        )
        nms = DetectionCandidate(
            raw_candidate_id=raw.raw_candidate_id,
            image_id=record.image_id,
            category="car",
            box_xyxy=raw.box_xyxy,
            confidence=raw.confidence,
            pre_nms=False,
            nms_cluster_id=raw.nms_cluster_id,
            source="test",
        )
        return [raw], [nms]

    monkeypatch.setattr(UltralyticsRawCandidateAdapter, "predict", synthetic_predict)
    report = cache_detector_stage(
        str(records),
        str(tmp_path / "detector_opts"),
        "ultralytics",
        weights="dummy.pt",
        confidence=0.01,
        imgsz=320,
        device="cpu",
        allow_post_nms_fallback=True,
    )

    assert report["status"] == "PASS"
    assert report["confidence"] == 0.01
    assert report["imgsz"] == 320
    assert report["device"] == "cpu"
    assert report["allow_post_nms_fallback"] is True


def test_cache_detector_can_record_detector_misses(tmp_path, fixture_source, monkeypatch):
    build = build_data_stage(
        fixture_source["annotations"], fixture_source["images_dir"], str(tmp_path / "data_miss")
    )
    assert build["status"] == "PASS"
    records = tmp_path / "data_miss" / "task_records.jsonl"

    def miss_predict(self, record):
        raise RawCandidateExtractionUnsupported("forced detector miss")

    monkeypatch.setattr(UltralyticsRawCandidateAdapter, "predict", miss_predict)
    report = cache_detector_stage(
        str(records),
        str(tmp_path / "detector_miss"),
        "ultralytics",
        weights="dummy.pt",
        detector_miss_policy="record",
    )

    assert report["status"] == "PASS"
    assert report["detector_miss_count"] == report["record_count"]
    assert report["candidate_covered_count"] == 0
    assert report["raw_candidate_count"] == 0
    assert report["nms_candidate_count"] == 0
    assert not report["detector_failures"]
    miss_rows = read_jsonl(tmp_path / "detector_miss" / "detector_misses.jsonl")
    assert len(miss_rows) == report["record_count"]
    assert (tmp_path / "detector_miss" / "detector_raw_candidates.jsonl").exists()
    assert (tmp_path / "detector_miss" / "detector_nms_candidates.jsonl").exists()


def test_train_cef_accepts_explicit_detector_miss_file(tmp_path, fixture_source, monkeypatch):
    build = build_data_stage(
        fixture_source["annotations"], fixture_source["images_dir"], str(tmp_path / "data_mixed")
    )
    assert build["status"] == "PASS"
    records = tmp_path / "data_mixed" / "task_records.jsonl"
    rows = read_jsonl(records)
    miss_row = next((row for row in rows if row.get("split") != "train"), rows[-1])
    missed_image_id = miss_row["image_id"]

    def mixed_predict(self, record):
        if record.image_id == missed_image_id:
            raise RawCandidateExtractionUnsupported("forced detector miss")
        raw = DetectionCandidate(
            raw_candidate_id=f"{record.image_id}:raw:0",
            image_id=record.image_id,
            category="car",
            box_xyxy=Box(1, 2, 10, 20),
            confidence=0.7,
            pre_nms=True,
            nms_cluster_id=f"{record.image_id}:cluster:0",
            source="test",
        )
        nms = DetectionCandidate(
            raw_candidate_id=raw.raw_candidate_id,
            image_id=record.image_id,
            category="car",
            box_xyxy=raw.box_xyxy,
            confidence=raw.confidence,
            pre_nms=False,
            nms_cluster_id=raw.nms_cluster_id,
            source="test",
        )
        return [raw], [nms]

    monkeypatch.setattr(UltralyticsRawCandidateAdapter, "predict", mixed_predict)
    detector = cache_detector_stage(
        str(records),
        str(tmp_path / "detector_mixed"),
        "ultralytics",
        weights="dummy.pt",
        detector_miss_policy="record",
    )
    assert detector["status"] == "PASS"
    assert detector["detector_miss_count"] == 1

    cef = train_cef_stage(
        str(records),
        str(tmp_path / "detector_mixed" / "detector_raw_candidates.jsonl"),
        str(tmp_path / "detector_mixed" / "detector_nms_candidates.jsonl"),
        str(tmp_path / "cef_mixed"),
        detector_misses_path=str(tmp_path / "detector_mixed" / "detector_misses.jsonl"),
    )
    assert cef["stage_execution_status"] == "PASS"
    assert cef["status"] in {"PASS", "HOLD"}
    assert cef["detector_miss_excluded_count"] == 1
    assert cef["usable_record_count"] == len(rows) - 1
    assert cef["pre_nms_cache_validation"]["status"] == "PASS"


def test_cache_detector_retries_miss_at_larger_imgsz(tmp_path, fixture_source, monkeypatch):
    build = build_data_stage(
        fixture_source["annotations"], fixture_source["images_dir"], str(tmp_path / "data_retry")
    )
    assert build["status"] == "PASS"
    records = tmp_path / "data_retry" / "task_records.jsonl"

    def retry_predict(self, record):
        if self.imgsz != 1280:
            raise RawCandidateExtractionUnsupported("primary detector miss")
        raw = DetectionCandidate(
            raw_candidate_id=f"{record.image_id}:raw:0",
            image_id=record.image_id,
            category="car",
            box_xyxy=Box(1, 2, 10, 20),
            confidence=0.7,
            pre_nms=True,
            nms_cluster_id=f"{record.image_id}:cluster:0",
            source="retry-test",
        )
        nms = DetectionCandidate(
            raw_candidate_id=raw.raw_candidate_id,
            image_id=record.image_id,
            category="car",
            box_xyxy=raw.box_xyxy,
            confidence=raw.confidence,
            pre_nms=False,
            nms_cluster_id=raw.nms_cluster_id,
            source="retry-test",
        )
        return [raw], [nms]

    monkeypatch.setattr(UltralyticsRawCandidateAdapter, "predict", retry_predict)
    report = cache_detector_stage(
        str(records),
        str(tmp_path / "detector_retry"),
        "ultralytics",
        weights="dummy.pt",
        confidence=0.001,
        imgsz=960,
        detector_miss_policy="record",
        miss_retry_imgsz=1280,
    )

    assert report["status"] == "PASS"
    assert report["detector_retry_count"] == report["record_count"]
    assert report["detector_miss_count"] == 0
    assert report["candidate_covered_count"] == report["record_count"]
