from visualact_r.data.splits import validate_split_integrity


def test_no_split_overlap(records):
    status = validate_split_integrity(records)
    assert status["status"] == "PASS"
    duplicate = [x for x in records if x.image_id in {"fixture_000.png", "fixture_001.png"}]
    assert len({x.split for x in duplicate}) == 1
