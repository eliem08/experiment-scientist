"""Project-aware summary adapters for NeuronLab result artifacts.

These are descriptive summaries of existing outputs, not validated model scores.
"""
from __future__ import annotations

from typing import Any


def summarize_artifact(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    """Return human-readable metrics supported by the artifact's own structure."""
    payload = artifact.get("payload", {})
    project = str(artifact.get("project", payload.get("manifest", {}).get("project", ""))).casefold()
    metrics: list[dict[str, Any]] = []

    def add(name: str, value: Any, note: str = "") -> None:
        if value is not None:
            metrics.append({"metric": name, "value": value, "note": note})

    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    cases = payload.get("cases")
    results = payload.get("results")
    if not isinstance(results, list):
        results = cases if isinstance(cases, list) else []

    if "neuronlab-workflow-comparison" in project or payload.get("schema_version") == "neuronlab.workflow-comparison.v1":
        gate = payload.get("gate") if isinstance(payload.get("gate"), dict) else {}
        add("Comparison gate", "PASS" if gate.get("passed") is True else "FAIL" if gate.get("passed") is False else None)
        baseline = payload.get("baseline") if isinstance(payload.get("baseline"), dict) else {}
        candidate = payload.get("candidate") if isinstance(payload.get("candidate"), dict) else {}
        add("Baseline model", baseline.get("model_id"))
        add("Candidate model", candidate.get("model_id"))
        comparisons = payload.get("comparisons", [])
        comparisons = [item for item in comparisons if isinstance(item, dict)] if isinstance(comparisons, list) else []
        for item in comparisons:
            delta = item.get("delta")
            if delta is not None:
                note = "Diagnostic only; does not affect the gate." if item.get("direction") == "neutral" or item.get("stage") == "safety-probe-lab" else ""
                add(f"{item.get('label', item.get('metric', 'Metric'))} delta", delta, note)
        blockers = gate.get("blockers", [])
        if blockers:
            add("Regression blockers", ", ".join(str(item) for item in blockers))
    elif "neuronlab-unified-workflow" in project or payload.get("schema_version") == "neuronlab.workflow.v1":
        stages = payload.get("stages", [])
        stages = [stage for stage in stages if isinstance(stage, dict)] if isinstance(stages, list) else []
        gate = payload.get("quality_gate")
        add("Workflow status", payload.get("status"))
        add("Stages", len(stages))
        add("Stages completed", sum(stage.get("status") == "completed" for stage in stages))
        add("Stages failed", sum(stage.get("status") == "failed" for stage in stages))
        if isinstance(gate, dict) and gate.get("passed") is not None:
            add("Unified quality gate", "PASS" if gate["passed"] else "FAIL")
        blockers = gate.get("blockers", []) if isinstance(gate, dict) else []
        if blockers:
            add("Gate blockers", ", ".join(str(item) for item in blockers))
        workflow_summary = payload.get("summary", {})
        if isinstance(workflow_summary, dict):
            behavior_metrics = workflow_summary.get("behavior_bench", {})
            safety_metrics = workflow_summary.get("safety_probe_lab", {})
            experiment_metrics = workflow_summary.get("experiment_scientist", {})
            if isinstance(behavior_metrics, dict):
                add("Behavior expected-check pass rate", behavior_metrics.get("expected_pass_rate"), "Configured substring checks, not semantic evaluation.")
                add("Behavior forbidden-hit cases", behavior_metrics.get("cases_with_forbidden_hits"))
            if isinstance(safety_metrics, dict):
                add("Safety heuristic signals", safety_metrics.get("heuristic_signals"), "Marker-based triage only.")
            if isinstance(experiment_metrics, dict):
                add("Experiment mean word delta", experiment_metrics.get("mean_word_delta"), "Descriptive length difference, not answer quality.")
    elif "behavior-bench" in project or "behavior_bench" in project:
        add("Cases evaluated", summary.get("total", len(results)))
        add("Expected-check pass rate", summary.get("expected_pass_rate"), "Fraction from configured substring checks; not semantic quality.")
        add("Cases with forbidden hits", summary.get("cases_with_forbidden_hits"))
        gate = payload.get("quality_gate")
        if not isinstance(gate, dict):
            gate = summary.get("quality_gate")
        add("Quality gate", "PASS" if gate.get("passed") is True else "FAIL" if gate.get("passed") is False else None,
            "Configured regression gate only; not a general quality certification.")
        structured = [row for row in results if isinstance(row, dict) and row.get("structured_pass") is not None]
        if structured:
            add("Structured JSON pass rate", sum(bool(row["structured_pass"]) for row in structured) / len(structured))
    elif "safety-probe-lab" in project or "safety_probe_lab" in project:
        evaluated = [row for row in results if isinstance(row, dict)]
        signals = [row for row in evaluated if row.get("heuristic_signal") is True]
        add("Probes evaluated", len(evaluated))
        add("Heuristic signal count", len(signals), "Marker-based triage only; not a validated safety score.")
        if evaluated:
            add("Heuristic signal rate", len(signals) / len(evaluated), "Interpret with the probe lab's limitations.")
        marker_counts = [row["marker_count"] for row in evaluated if isinstance(row.get("marker_count"), (int, float))]
        if marker_counts:
            add("Mean marker count", sum(marker_counts) / len(marker_counts))
    elif "experiment-scientist" in project or "experiment_scientist" in project:
        pairs = [row for row in results if isinstance(row, dict) and isinstance(row.get("word_delta_treatment_minus_baseline"), (int, float))]
        bootstrap = payload.get("bootstrap_word_delta")
        add("Paired prompts", len(results))
        if pairs:
            add("Mean treatment-minus-baseline word delta", sum(row["word_delta_treatment_minus_baseline"] for row in pairs) / len(pairs),
                "Descriptive length difference, not a measure of answer quality.")
        if isinstance(bootstrap, dict):
            add("Bootstrap mean word delta", bootstrap.get("mean"), "Exploratory; small prompt sets yield uncertain intervals.")
            add("Bootstrap CI lower", bootstrap.get("lower"))
            add("Bootstrap CI upper", bootstrap.get("upper"))

    # Keep useful tool-specific summaries first, then provide generic top-level
    # numeric fields when no project adapter has a meaningful value.
    if not metrics:
        for key, value in summary.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                add(key.replace("_", " ").title(), value)
    return metrics


def compare_metric_maps(metric_maps: list[dict[str, int | float]], run_names: list[str]) -> list[dict[str, Any]]:
    """Build a common-metric comparison with absolute deltas versus the first run."""
    if len(metric_maps) != len(run_names) or len(metric_maps) < 2:
        return []
    common = sorted(set.intersection(*(set(metrics) for metrics in metric_maps)))
    rows = []
    baseline_name = run_names[0]
    for metric in common:
        row: dict[str, Any] = {"metric": metric}
        values = [mapping[metric] for mapping in metric_maps]
        row[baseline_name] = values[0]
        for index in range(1, len(metric_maps)):
            row[run_names[index]] = values[index]
            row[f"delta vs {baseline_name} ({run_names[index]})"] = values[index] - values[0]
        rows.append(row)
    return rows
