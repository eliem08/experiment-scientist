"""Small, dependency-free statistical utilities for paired model evaluations.

The independent unit is explicit: repeated observations are averaged within each
pairing key before case-level inference. These utilities are deliberately modest;
they do not replace a preregistered analysis plan or expert statistical review.
"""
from __future__ import annotations

import itertools
import math
import random
import statistics
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence


def paired_unit_deltas(
    observations: Iterable[Mapping[str, Any]], *,
    pair_key: str = "case_id", baseline_key: str = "baseline_score",
    treatment_key: str = "treatment_score",
) -> list[dict[str, Any]]:
    """Aggregate repeated rows within independent pair IDs before inference.

    Each input row must contain numeric baseline and treatment scores. Multiple
    rows per pair (e.g. repeated seeds) are averaged separately by arm; the result
    has one treatment-minus-baseline delta per pair. Missing or non-finite scores
    are rejected rather than silently imputed.
    """
    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"baseline": [], "treatment": []})
    for index, row in enumerate(observations):
        pair_id = row.get(pair_key)
        if not isinstance(pair_id, (str, int)) or not str(pair_id).strip():
            raise ValueError(f"observation {index} has no valid pairing key {pair_key!r}")
        baseline, treatment = row.get(baseline_key), row.get(treatment_key)
        for name, value in (("baseline", baseline), ("treatment", treatment)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"observation {index} has invalid {name} score")
            groups[str(pair_id)][name].append(float(value))
    result = []
    for pair_id in sorted(groups):
        group = groups[pair_id]
        if not group["baseline"] or not group["treatment"]:
            raise ValueError(f"pair {pair_id!r} is missing an arm")
        baseline_mean = statistics.fmean(group["baseline"])
        treatment_mean = statistics.fmean(group["treatment"])
        result.append({
            "pair_id": pair_id,
            "n_baseline": len(group["baseline"]),
            "n_treatment": len(group["treatment"]),
            "baseline_mean": baseline_mean,
            "treatment_mean": treatment_mean,
            "delta": treatment_mean - baseline_mean,
        })
    return result


def mean_confidence_interval(
    values: Sequence[float], *, confidence: float = 0.95,
    iterations: int = 5000, seed: int = 17,
) -> dict[str, Any]:
    """Percentile bootstrap interval for the mean of independent-unit values."""
    _validate_values(values)
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between 0 and 1")
    if iterations < 100:
        raise ValueError("iterations must be at least 100")
    if not values:
        return {"n": 0, "mean": None, "lower": None, "upper": None,
                "confidence": confidence, "method": "percentile bootstrap"}
    mean = statistics.fmean(values)
    if len(values) == 1:
        return {"n": 1, "mean": mean, "lower": None, "upper": None,
                "confidence": confidence, "method": "not estimable for n < 2"}
    rng = random.Random(seed)
    boot_means = sorted(
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(iterations)
    )
    alpha = (1.0 - confidence) / 2.0
    low_index = max(0, min(iterations - 1, math.floor(alpha * iterations)))
    high_index = max(0, min(iterations - 1, math.ceil((1.0 - alpha) * iterations) - 1))
    return {
        "n": len(values), "mean": mean,
        "lower": boot_means[low_index], "upper": boot_means[high_index],
        "confidence": confidence, "iterations": iterations,
        "method": "percentile bootstrap over independent paired units",
    }


def paired_effect_size(values: Sequence[float]) -> dict[str, Any]:
    """Return paired-delta SD and Cohen's dz where estimable."""
    _validate_values(values)
    if len(values) < 2:
        return {"n": len(values), "mean_delta": statistics.fmean(values) if values else None,
                "sd_delta": None, "cohens_dz": None}
    sd = statistics.stdev(values)
    mean = statistics.fmean(values)
    return {"n": len(values), "mean_delta": mean, "sd_delta": sd,
            "cohens_dz": mean / sd if sd > 0 else None}


def sign_flip_permutation_test(
    values: Sequence[float], *, iterations: int = 20000, seed: int = 17,
    exact_max_n: int = 16,
) -> dict[str, Any]:
    """Two-sided sign-flip test for paired deltas, with exact enumeration for small n.

    This test assumes exchangeability of signs under the null. It is not suitable
    when pairing or the experimental design does not justify that assumption.
    """
    _validate_values(values)
    if iterations < 100:
        raise ValueError("iterations must be at least 100")
    if exact_max_n < 1:
        raise ValueError("exact_max_n must be positive")
    n = len(values)
    if n == 0:
        return {"n": 0, "observed_mean": None, "p_value": None, "method": "not estimable"}
    observed = abs(statistics.fmean(values))
    tolerance = 1e-12
    if n <= exact_max_n:
        means = (
            abs(sum(value * sign for value, sign in zip(values, signs)) / n)
            for signs in itertools.product((-1.0, 1.0), repeat=n)
        )
        count = sum(value >= observed - tolerance for value in means)
        total = 2 ** n
        method = "exact sign-flip enumeration"
    else:
        rng = random.Random(seed)
        count = 0
        for _ in range(iterations):
            randomized = abs(statistics.fmean(value * rng.choice((-1.0, 1.0)) for value in values))
            count += randomized >= observed - tolerance
        # Add-one correction avoids a Monte Carlo estimate of exactly zero.
        total = iterations + 1
        count += 1
        method = "Monte Carlo sign-flip permutation with add-one correction"
    return {"n": n, "observed_mean": statistics.fmean(values),
            "p_value": count / total, "method": method,
            "iterations": total if method.startswith("Monte Carlo") else None,
            "assumption": "Sign exchangeability under the null is required."}


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values, preserving input keys and order."""
    checked: list[tuple[str, float]] = []
    for key, value in p_values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0 <= value <= 1:
            raise ValueError(f"p-value for {key!r} must be finite and between 0 and 1")
        checked.append((key, float(value)))
    ordered = sorted(checked, key=lambda item: item[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (key, p_value) in enumerate(ordered):
        candidate = min(1.0, (m - rank) * p_value)
        running = max(running, candidate)
        adjusted[key] = running
    return {key: adjusted[key] for key, _ in checked}


def _validate_values(values: Sequence[float]) -> None:
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"value at index {index} must be a finite number")
