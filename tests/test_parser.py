import pytest
from visualact_r.agents.core import parse_typed_output


def test_typed_parser():
    assert (
        parse_typed_output(
            '{"tool":"FINALIZE","decision":"REVIEW","confidence":0.5,"evidence_ids":[],"reason_code":"test"}'
        )["decision"]
        == "REVIEW"
    )
    with pytest.raises(ValueError):
        parse_typed_output('{"tool":"FINALIZE","decision":"BAD"}')
