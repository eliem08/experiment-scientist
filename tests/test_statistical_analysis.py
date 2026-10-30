import pytest

from statistical_analysis import (
    holm_adjust,
    mean_confidence_interval,
    paired_effect_size,
    paired_unit_deltas,
    sign_flip_permutation_test,
)


def test_repeated_observations_are_aggregated_by_independent_case():
    rows = [
        {"case_id": "b", "baseline_score": 0, "treatment_score": 1},
        {"case_id": "a", "baseline_score": 1, "treatment_score": 2},
        {"case_id": "a", "baseline_score": 3, "treatment_score": 5},
    ]
    pairs = paired_unit_deltas(rows)
    assert [row["pair_id"] for row in pairs] == ["a", "b"]
    assert pairs[0]["baseline_mean"] == 2
    assert pairs[0]["treatment_mean"] == 3.5
    assert pairs[0]["delta"] == 1.5


def test_paired_rows_reject_invalid_values_and_missing_pair_ids():
    with pytest.raises(ValueError, match="pairing key"):
        paired_unit_deltas([{"baseline_score": 1, "treatment_score": 2}])
    with pytest.raises(ValueError, match="invalid baseline"):
        paired_unit_deltas([{"case_id": "x", "baseline_score": float("nan"), "treatment_score": 2}])


def test_bootstrap_is_reproducible_and_reports_single_case_limit():
    first = mean_confidence_interval([1.0, 2.0, 3.0], seed=9)
    assert first == mean_confidence_interval([1.0, 2.0, 3.0], seed=9)
    one = mean_confidence_interval([2.0])
    assert one["lower"] is None and one["upper"] is None
    assert mean_confidence_interval([])["n"] == 0


def test_effect_size_handles_zero_variance():
    assert paired_effect_size([1, 1, 1])["cohens_dz"] is None
    assert paired_effect_size([1, 2, 3])["cohens_dz"] == pytest.approx(2.0)


def test_sign_flip_exact_and_monte_carlo_modes_are_valid():
    exact = sign_flip_permutation_test([1, 1, 1])
    assert exact["method"] == "exact sign-flip enumeration"
    assert exact["p_value"] == pytest.approx(0.25)
    monte_carlo = sign_flip_permutation_test([1, 2, 3] * 6, iterations=1000, seed=2, exact_max_n=8)
    assert monte_carlo["method"].startswith("Monte Carlo")
    assert 0 < monte_carlo["p_value"] <= 1


def test_holm_adjustment_is_monotonic_and_preserves_keys():
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == {"a": 0.03, "b": 0.06, "c": 0.06}
    with pytest.raises(ValueError, match="p-value"):
        holm_adjust({"bad": 1.1})


def test_paired_repeats_report_equal_arm_counts():
    rows = [
        {"case_id": "x", "baseline_score": 1, "treatment_score": 2},
        {"case_id": "x", "baseline_score": 2, "treatment_score": 3},
        {"case_id": "x", "baseline_score": 3, "treatment_score": 4},
    ]
    pair = paired_unit_deltas(rows)[0]
    assert pair["n_baseline"] == 3
    assert pair["n_treatment"] == 3
    assert pair["delta"] == pytest.approx(1.0)
