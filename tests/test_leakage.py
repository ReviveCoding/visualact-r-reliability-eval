def test_inference_contract_has_no_gt(records):
    payload = records[0].inference_contract()
    assert "gt_objects" not in payload
    assert "action_label" not in payload
    assert "supporting_object_ids" not in payload
