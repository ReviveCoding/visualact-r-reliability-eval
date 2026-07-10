from visualact_r.cef.core import fit_cef, make_crops
from visualact_r.detector.synthetic import SyntheticColorDetector


def test_cef_crops_are_bounded(records, config):
    detector = SyntheticColorDetector()
    cache = {x.image_id: detector.predict(x) for x in records}
    train = [x for x in records if x.split == "train"]
    model = fit_cef(train, cache, config)
    target = train[0]
    raw, nms = cache[target.image_id]
    crops = make_crops(target, raw, nms, "cef", config, model)
    assert all(0 <= c.box_xyxy.x1 <= c.box_xyxy.x2 <= target.width for c in crops)
    assert all(0 <= c.box_xyxy.y1 <= c.box_xyxy.y2 <= target.height for c in crops)
    assert all(c.raw_candidate_ids for c in crops)
