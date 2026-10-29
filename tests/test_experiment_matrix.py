import json

import pytest

from experiment_matrix import (
    SPEC_VERSION,
    bootstrap_mean_ci,
    load_and_validate_spec,
    score_response,
    summarize_pairs,
)


def _spec(tmp_path, **overrides):
    payload = {
        "schema_version": SPEC_VERSION,
        "title": "Test experiment",
        "hypothesis": "Treatment increases explicit uncertainty.",
        "primary_metric": "uncertainty",
        "metrics": [
            {"id": "uncertainty", "kind": "contains", "value": "I don't know", "direction": "higher"},
            {"id": "length", "kind": "word_count", "direction": "neutral"},
            {"id": "regex", "kind": "regex_match", "pattern": r"\b\d{4}\b", "direction": "higher"},
            {"id": "exact", "kind": "exact_match", "value": "yes", "direction": "higher"},
            {"id": "chars", "kind": "character_count", "direction": "lower"},
        ],
        "cases": [{"id": "one", "baseline": "Question?", "treatment": "Question with detail?"}],
    }
    payload.update(overrides)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_validates_spec_and_returns_exact_file_hash(tmp_path):
    path = _spec(tmp_path)
    spec, fingerprint = load_and_validate_spec(path)
    assert spec["title"] == "Test experiment"
    assert len(fingerprint) == 64
    assert fingerprint == load_and_validate_spec(path)[1]


@pytest.mark.parametrize("overrides, message", [
    ({"schema_version": "wrong"}, "schema_version"),
    ({"primary_metric": "missing"}, "primary_metric"),
    ({"metrics": []}, "metrics"),
    ({"cases": []}, "cases"),
    ({"metrics": [{"id": "x", "kind": "mystery"}]}, "Unsupported metric"),
    ({"metrics": [{"id": "x", "kind": "contains"}]}, "requires a string"),
    ({"metrics": [{"id": "x", "kind": "regex_match", "pattern": "["}]}, "Invalid regex"),
    ({"cases": [{"id": "same", "baseline": "a", "treatment": "b"}, {"id": "same", "baseline": "c", "treatment": "d"}]}, "unique"),
])
def test_rejects_invalid_spec(tmp_path, overrides, message):
    path = _spec(tmp_path, **overrides)
    with pytest.raises(ValueError, match=message):
        load_and_validate_spec(path)


def test_metric_scorers_are_deterministic_and_case_insensitive_where_expected():
    assert score_response("I DON'T KNOW", {"kind": "contains", "value": "I don't know"}) == 1.0
    assert score_response("No idea", {"kind": "contains", "value": "I don't know"}) == 0.0
    assert score_response("a b c", {"kind": "word_count"}) == 3.0
    assert score_response("abcd", {"kind": "character_count"}) == 4.0
    assert score_response("answer 2026", {"kind": "regex_match", "pattern": r"\b\d{4}\b"}) == 1.0
    assert score_response("yes ", {"kind": "exact_match", "value": "yes"}) == 1.0


def test_bootstrap_is_seeded_and_reports_small_sample_limitations():
    assert bootstrap_mean_ci([1.0, 2.0, 3.0], seed=9) == bootstrap_mean_ci([1.0, 2.0, 3.0], seed=9)
    one = bootstrap_mean_ci([2.0])
    assert one["mean"] == 2.0
    assert one["lower"] is None and one["upper"] is None
    assert summarize_pairs([1.0, 2.0, 3.0], seed=17)["cohens_dz"] is not None


def test_bootstrap_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="iterations"):
        bootstrap_mean_ci([1.0, 2.0], iterations=10)
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_mean_ci([1.0, 2.0], confidence=1.0)
