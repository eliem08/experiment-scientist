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
    ({"metrics": [{"id": "uncertainty", "kind": "mystery"}]}, "Unsupported metric"),
    ({"metrics": [{"id": "uncertainty", "kind": "contains"}]}, "requires a string"),
    ({"metrics": [{"id": "uncertainty", "kind": "regex_match", "pattern": "["}]}, "Invalid regex"),
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


def test_checkpoint_only_loads_for_matching_configuration(tmp_path):
    from experiment_matrix import load_checkpoint_observations

    path = tmp_path / "matrix-checkpoint.json"
    path.write_text(json.dumps({
        "schema_version": "experiment-scientist.checkpoint.v1",
        "config_sha256": "a" * 64,
        "observations": [{"model_id": "m", "seed": 17, "case_id": "c", "metric_id": "x"}],
    }), encoding="utf-8")
    assert load_checkpoint_observations(path, "a" * 64)[0]["metric_id"] == "x"
    with pytest.raises(ValueError, match="does not match"):
        load_checkpoint_observations(path, "b" * 64)


def test_checkpoint_rejects_wrong_schema_and_invalid_observations(tmp_path):
    from experiment_matrix import load_checkpoint_observations

    path = tmp_path / "matrix-checkpoint.json"
    path.write_text(json.dumps({"config_sha256": "a" * 64, "observations": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_checkpoint_observations(path, "a" * 64)
    path.write_text(json.dumps({
        "schema_version": "experiment-scientist.checkpoint.v1",
        "config_sha256": "a" * 64,
        "observations": "not-a-list",
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="list of objects"):
        load_checkpoint_observations(path, "a" * 64)


def test_resume_skips_only_complete_pairs_and_rejects_partial_pairs():
    from experiment_matrix import pair_is_complete

    observations = [
        {"model_id": "m", "seed": 17, "case_id": "c", "metric_id": "a"},
        {"model_id": "m", "seed": 17, "case_id": "c", "metric_id": "b"},
    ]
    assert pair_is_complete(observations, model_id="m", seed=17, case_id="c", metric_ids={"a", "b"})
    assert not pair_is_complete(observations, model_id="m", seed=29, case_id="c", metric_ids={"a", "b"})
    with pytest.raises(ValueError, match="Incomplete or duplicate"):
        pair_is_complete(observations[:1], model_id="m", seed=17, case_id="c", metric_ids={"a", "b"})
    with pytest.raises(ValueError, match="Incomplete or duplicate"):
        pair_is_complete(observations + [observations[0]], model_id="m", seed=17, case_id="c", metric_ids={"a", "b"})
