from visualact_r.agents.core import MockVisualAgent
from visualact_r.cef.core import fit_cef, make_crops
from visualact_r.detector.synthetic import SyntheticColorDetector
from visualact_r.rctr.core import fit_rctr


def test_rctr_train_only_references(records, config):
    detector = SyntheticColorDetector()
    cache = {x.image_id: detector.predict(x) for x in records}
    train = [x for x in records if x.split == "train"]
    cef = fit_cef(train, cache, config)
    crops = {}
    traces = []
    for record in train:
        raw, nms = cache[record.image_id]
        crops[record.image_id] = make_crops(record, raw, nms, "cef", config, cef)
        traces.append(MockVisualAgent("C5").rollout(record, crops[record.image_id], config))
    model = fit_rctr(train, traces, crops)
    assert all(
        trace_id.split(":")[0] in {x.image_id for x in train}
        for trace_id in model.reference_trace_ids
    )
