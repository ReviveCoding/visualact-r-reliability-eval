from visualact_r.schemas import Box, box_iou


def test_box_iou_and_clamp():
    assert box_iou(Box(0, 0, 10, 10), Box(5, 5, 15, 15)) == 25 / 175
    assert Box(-2, -1, 12, 13).clamp(10, 10) == Box(0, 0, 10, 10)
