from visualact_r.pipeline import run_fixture_pipeline


def test_fixture_e2e(tmp_path):
    result = run_fixture_pipeline(tmp_path / "run")
    assert result["release"]["status"] == "FIXTURE_ONLY_NO_PERFORMANCE_CLAIM"
    for name in [
        "metrics.json",
        "candidate_comparison.csv",
        "slice_metrics.csv",
        "decision_scenarios.csv",
        "operational_metrics.json",
        "run_manifest.json",
        "report.md",
        "release_decision.json",
        "project_creation_handoff.json",
    ]:
        assert (tmp_path / "run" / name).exists()
