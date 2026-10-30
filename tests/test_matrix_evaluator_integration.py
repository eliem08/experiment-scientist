import pytest

from experiment_matrix import SPEC_VERSION, load_and_validate_spec, score_response


def _spec(tmp_path, metric):
    payload = {
        "schema_version": SPEC_VERSION,
        "title": "Structured evaluator test",
        "hypothesis": "Structured answers match the reference.",
        "primary_metric": metric["id"],
        "metrics": [metric],
        "cases": [{"id": "case-1", "baseline": "Answer:", "treatment": "Answer with JSON:"}],
    }
    path = tmp_path / "spec.json"
    import json
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_matrix_reuses_shared_evaluator_core_for_existing_metric_types():
    assert score_response("Hello there", {"kind": "contains", "value": "hello"}) == 1.0
    assert score_response(" exact ", {"kind": "exact_match", "value": "exact"}) == 1.0
    assert score_response("Year 2026", {"kind": "regex_match", "pattern": r"\b\d{4}\b"}) == 1.0


def test_matrix_supports_structured_json_field_evaluator(tmp_path):
    metric = {
        "id": "answer",
        "kind": "json_field_equals",
        "field_path": "result.answer",
        "reference": {"result": {"answer": "42"}},
    }
    load_and_validate_spec(_spec(tmp_path, metric))
    assert score_response('{"result":{"answer":"42"}}', metric) == 1.0
    assert score_response('{"result":{"answer":"41"}}', metric) == 0.0
    assert score_response("not json", metric) == 0.0


@pytest.mark.parametrize("metric, message", [
    ({"id": "answer", "kind": "json_field_equals", "reference": {"result": 1}}, "field_path"),
    ({"id": "answer", "kind": "json_field_equals", "field_path": "result.value"}, "reference"),
])
def test_matrix_rejects_incomplete_json_field_metric(tmp_path, metric, message):
    with pytest.raises(ValueError, match=message):
        load_and_validate_spec(_spec(tmp_path, metric))
