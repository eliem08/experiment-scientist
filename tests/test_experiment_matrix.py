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


def test_report_renderer_includes_outcomes_performance_and_reproducibility():
    from render_matrix_report import render_html, render_markdown

    payload = {
        "schema_version": "experiment-scientist.matrix.v1",
        "experiment": {"title": "Ablation", "hypothesis": "Treatment helps", "primary_metric": "contains", "spec_sha256": "abc"},
        "models": ["toy-model"],
        "device": "cpu",
        "seeds": [17, 29],
        "runtime_seconds": 12.5,
        "summaries": [{
            "model_id": "toy-model", "metric_id": "contains", "direction": "higher",
            "cases": 4, "seeds_per_case": 2, "mean": 0.5, "lower": 0.1, "upper": 0.9, "cohens_dz": 0.7,
        }],
        "performance_summary": [{
            "model_id": "toy-model", "paired_cases": 8,
            "baseline_mean_generation_seconds": 0.4, "treatment_mean_generation_seconds": 0.5,
            "baseline_mean_output_tokens_per_second": 20.0, "treatment_mean_output_tokens_per_second": 18.0,
        }],
        "reproducibility": {"python_version": "3.11", "torch_version": "2.x", "github_sha": "deadbeef"},
        "model_revisions": {"toy-model": {"resolved_commit": "modelsha", "tokenizer_resolved_commit": "toksha"}},
        "interpretation_limits": ["This is a rule-based metric."],
        "observations": [],
    }
    markdown = render_markdown(payload)
    assert "Outcome summaries" in markdown
    assert "Performance" in markdown
    assert "0.4000" in markdown
    assert "modelsha" in markdown
    assert "rule-based metric" in markdown
    page = render_html(payload)
    assert "<!doctype html>" in page
    assert "&lt;script&gt;" not in page


def test_report_renderer_escapes_untrusted_html_fields():
    from render_matrix_report import render_html

    payload = {
        "experiment": {"title": "<script>alert(1)</script>", "hypothesis": "<img src=x>"},
        "models": [], "seeds": [], "summaries": [], "performance_summary": [],
        "reproducibility": {}, "model_revisions": {},
        "interpretation_limits": [], "observations": [],
    }
    page = render_html(payload)
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
