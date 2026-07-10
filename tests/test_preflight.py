from visualact_r.data.preflight import preflight_fixture


def test_fixture_preflight(fixture_source, tmp_path):
    report = preflight_fixture(fixture_source["root"], tmp_path / "out")
    assert report["status"] == "PASS"
    assert report["image_label_join_rate"] == 1.0
