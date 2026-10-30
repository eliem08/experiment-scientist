import json

import pytest

from evaluation_core import (
    ContainsEvaluator,
    EvaluationInput,
    ExactMatchEvaluator,
    JSONFieldEqualsEvaluator,
    LLMJudgeEvaluator,
    RegexMatchEvaluator,
    validate_human_annotation,
)


def test_exact_match_normalizes_only_configured_dimensions():
    item = EvaluationInput("case-1", "  Hello ", reference="hello")
    assert ExactMatchEvaluator(case_sensitive=False).evaluate(item).score == 1.0
    assert ExactMatchEvaluator(case_sensitive=True).evaluate(item).score == 0.0


def test_contains_metric_is_explicitly_lexical():
    result = ContainsEvaluator("I don't know").evaluate(
        EvaluationInput("case-1", "I DON'T KNOW the answer")
    )
    assert result.score == 1.0
    assert result.evaluator_kind == "rule_based"
    assert "not semantic correctness" in result.metadata.get("note", "") or result.evaluator_id == "contains"


def test_regex_rejects_invalid_patterns_and_scores_matches():
    with pytest.raises(ValueError, match="Invalid evaluator regex"):
        RegexMatchEvaluator("[")
    result = RegexMatchEvaluator(r"\b\d{4}\b").evaluate(
        EvaluationInput("year", "Published in 2026")
    )
    assert result.score == 1.0


def test_json_field_evaluator_handles_invalid_json_as_zero_not_exception():
    evaluator = JSONFieldEqualsEvaluator("answer.value")
    valid = evaluator.evaluate(EvaluationInput("a", '{"answer":{"value":42}}', reference={"answer":{"value":42}}))
    invalid = evaluator.evaluate(EvaluationInput("b", "not json", reference={"answer":{"value":42}}))
    assert valid.score == 1.0
    assert invalid.score == 0.0
    assert invalid.label == "invalid_or_missing"


def test_judge_accepts_valid_structured_decision_and_preserves_audit_metadata():
    evaluator = LLMJudgeEvaluator(
        judge_id="fixture-judge", judge_revision="abc123", rubric="Score answer quality.",
        judge_call=lambda prompt, context: {"score": 0.75, "label": "good", "rationale": "Meets rubric."},
    )
    result = evaluator.evaluate(EvaluationInput("case-1", "A response"))
    payload = result.to_dict()
    assert result.status == "completed"
    assert result.score == 0.75
    assert payload["raw_decision"]["label"] == "good"
    assert payload["metadata"]["judge_revision"] == "abc123"


def test_judge_retries_boundedly_then_marks_missing_not_zero():
    calls = []
    def invalid(prompt, context):
        calls.append(context["attempt"])
        return '{"score": 99, "label": "bad", "rationale": "out of range"}'

    result = LLMJudgeEvaluator(
        judge_id="fixture", judge_revision=None, rubric="Judge.", judge_call=invalid, max_retries=2
    ).evaluate(EvaluationInput("case", "response"))
    assert calls == [1, 2, 3]
    assert result.status == "failed"
    assert result.score is None
    assert any("missing data" in warning for warning in result.warnings)


def test_judge_handles_json_text_and_rejects_boolean_score():
    result = LLMJudgeEvaluator(
        judge_id="fixture", judge_revision=None, rubric="Judge.",
        judge_call=lambda p, c: json.dumps({"score": True, "label": "x", "rationale": "y"}),
        max_retries=0,
    ).evaluate(EvaluationInput("case", "response"))
    assert result.status == "failed"
    assert result.score is None


def test_human_annotation_requires_auditable_identity_and_rubric():
    annotation = validate_human_annotation({
        "case_id": "c1", "annotator_id": "annotator-1", "label": "correct", "rubric_version": "v1",
    })
    assert annotation["schema_version"] == "experiment-scientist.human-annotation.v1"
    assert annotation["created_at_utc"]
    with pytest.raises(ValueError, match="annotator_id"):
        validate_human_annotation({"case_id": "c1", "label": "correct", "rubric_version": "v1"})
