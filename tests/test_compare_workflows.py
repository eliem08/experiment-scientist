import pytest

from compare_workflows import compare_workflows


def _workflow(workflow_id, *, pass_rate=1.0, forbidden=0, signals=2, word_delta=3.0, gate=True, model="toy"):
    return {
        "schema_version": "neuronlab.workflow.v1",
        "workflow_id": workflow_id,
        "model_id": model,
        "config": {"seed": 17, "max_new_tokens": 32},
        "quality_gate": {"passed": gate, "blockers": [] if gate else ["example"]},
        "summary": {
            "behavior_bench": {
                "expected_pass_rate": pass_rate,
                "cases_with_forbidden_hits": forbidden,
            },
            "safety_probe_lab": {"heuristic_signals": signals},
            "experiment_scientist": {"mean_word_delta": word_delta},
        },
    }


def test_comparison_passes_when_configured_metrics_do_not_regress():
    result = compare_workflows(
        _workflow("baseline", pass_rate=0.9, forbidden=1),
        _workflow("candidate", pass_rate=0.88, forbidden=1),
        max_pass_rate_drop=0.05,
    )
    assert result["gate"]["passed"] is True
    assert result["comparison_id"]
    assert result["comparison_id"] != result["candidate"]["workflow_id"]
    assert result["comparisons"][0]["delta"] == pytest.approx(-0.02)


def test_comparison_blocks_pass_rate_drop_beyond_threshold():
    result = compare_workflows(
        _workflow("baseline", pass_rate=0.95),
        _workflow("candidate", pass_rate=0.80),
        max_pass_rate_drop=0.05,
    )
    assert result["gate"]["passed"] is False
    assert any("expected_pass_rate" in blocker for blocker in result["gate"]["blockers"])


def test_comparison_blocks_forbidden_hit_increase_by_default():
    result = compare_workflows(
        _workflow("baseline", forbidden=0),
        _workflow("candidate", forbidden=1),
    )
    assert result["gate"]["passed"] is False
    assert any("cases_with_forbidden_hits" in blocker for blocker in result["gate"]["blockers"])


def test_heuristic_counts_and_word_deltas_are_diagnostics_not_gate_metrics():
    result = compare_workflows(
        _workflow("baseline", signals=0, word_delta=1),
        _workflow("candidate", signals=100, word_delta=999),
    )
    assert result["gate"]["passed"] is True
    assert result["comparisons"][2]["delta"] == 100
    assert result["comparisons"][2]["regression"] is False
    assert result["comparisons"][3]["regression"] is False


def test_comparison_warns_when_configs_or_models_differ():
    baseline = _workflow("baseline", model="model-a")
    candidate = _workflow("candidate", model="model-b")
    candidate["config"]["seed"] = 99
    result = compare_workflows(baseline, candidate)
    assert any("configurations differ" in warning for warning in result["warnings"])
    assert any("Model IDs differ" in warning for warning in result["warnings"])


def test_comparison_fails_if_candidate_workflow_gate_failed():
    result = compare_workflows(
        _workflow("baseline"),
        _workflow("candidate", gate=False),
    )
    assert result["gate"]["passed"] is False
    assert any("candidate-workflow-quality-gate" in blocker for blocker in result["gate"]["blockers"])


def test_comparison_rejects_invalid_thresholds():
    with pytest.raises(ValueError, match="max_pass_rate_drop"):
        compare_workflows(_workflow("a"), _workflow("b"), max_pass_rate_drop=1.5)
    with pytest.raises(ValueError, match="max_forbidden_hit_increase"):
        compare_workflows(_workflow("a"), _workflow("b"), max_forbidden_hit_increase=-1)


def test_comparison_warns_when_case_suite_fingerprints_differ():
    baseline = _workflow("baseline")
    candidate = _workflow("candidate")
    baseline["reproducibility"] = {"case_suite": {"sha256": "a" * 64}}
    candidate["reproducibility"] = {"case_suite": {"sha256": "b" * 64}}
    result = compare_workflows(baseline, candidate)
    assert any("Case-suite SHA-256 fingerprints differ" in warning for warning in result["warnings"])



def test_comparison_fails_closed_when_required_regression_metric_is_missing():
    baseline = _workflow("baseline")
    candidate = _workflow("candidate")
    del candidate["summary"]["behavior-bench"]["expected_pass_rate"]
    result = compare_workflows(baseline, candidate)
    assert result["gate"]["passed"] is False
    assert any("required regression metric is missing" in blocker for blocker in result["gate"]["blockers"])
