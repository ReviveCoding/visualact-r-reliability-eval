from visualact_r.data.task_policy import triage_policy
from visualact_r.schemas import Box, GroundTruthObject


def test_primary_large_object_escalates(config):
    objects = [GroundTruthObject("a", "car", Box(280, 210, 400, 350))]
    action, support, flags = triage_policy(objects, 640, 360, config)
    assert action == "ESCALATE"
    assert support == ["a"]
    assert "primary_zone" in flags
