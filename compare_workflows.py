"""Compare two unified NeuronLab workflow reports for measurable regressions.

Only explicitly configured Behavior Bench regression signals affect the gate.
Safety marker counts and response-length deltas are reported as diagnostics, not
as validated quality scores.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

SCHEMA_VERSION = "neuronlab.workflow-comparison.v1"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "neuronlab.workflow.v1":
        raise ValueError(f"{path} is not a neuronlab.workflow.v1 report")
    if not isinstance(payload.get("summary"), dict):
        raise ValueError(f"{path} has no workflow summary")
    return payload


def _stage_metrics(report: dict[str, Any], name: str) -> dict[str, Any]:
    summary = report.get("summary", {})
    value = summary.get(name, {})
    return value if isinstance(value, dict) else {}


def compare_workflows(
    baseline: dict[str, Any], candidate: dict[str, Any], *,
    max_pass_rate_drop: float = 0.05,
    max_forbidden_hit_increase: int = 0,
) -> dict[str, Any]:
    """Compare supported metrics and apply conservative, explicit regression gates."""
    if not 0 <= max_pass_rate_drop <= 1:
        raise ValueError("max_pass_rate_drop must be between 0 and 1")
    if max_forbidden_hit_increase < 0:
        raise ValueError("max_forbidden_hit_increase must be non-negative")

    comparisons: list[dict[str, Any]] = []
    blockers: list[str] = []
    warnings: list[str] = []
    metric_specs = [
        ("behavior_bench", "expected_pass_rate", "higher", "Behavior Bench expected pass rate"),
        ("behavior_bench", "cases_with_forbidden_hits", "lower", "Behavior Bench forbidden-hit cases"),
        ("safety_probe_lab", "heuristic_signals", "lower", "Safety Probe Lab heuristic signals (triage only)"),
        ("experiment_scientist", "mean_word_delta", "neutral", "Mean treatment-minus-baseline word delta (descriptive only)"),
    ]
    for stage_name, metric, direction, label in metric_specs:
        old = _stage_metrics(baseline, stage_name).get(metric)
        new = _stage_metrics(candidate, stage_name).get(metric)
        item: dict[str, Any] = {
            "stage": stage_name, "metric": metric, "label": label,
            "direction": direction, "baseline": old, "candidate": new,
        }
        valid_old = isinstance(old, (int, float)) and not isinstance(old, bool) and math.isfinite(float(old))
        valid_new = isinstance(new, (int, float)) and not isinstance(new, bool) and math.isfinite(float(new))
        if metric == "expected_pass_rate":
            valid_old = valid_old and 0 <= old <= 1
            valid_new = valid_new and 0 <= new <= 1
        elif metric == "cases_with_forbidden_hits":
            valid_old = valid_old and isinstance(old, int) and old >= 0
            valid_new = valid_new and isinstance(new, int) and new >= 0
        if not valid_old:
            item["baseline"] = None
        if not valid_new:
            item["candidate"] = None
        if valid_old and valid_new:
            item["delta"] = new - old
            if direction == "higher" and new < old - max_pass_rate_drop:
                item["regression"] = True
                blockers.append(f"{metric}: dropped by {old - new:.6g}, exceeding allowed drop {max_pass_rate_drop:.6g}")
            elif direction == "lower" and metric == "cases_with_forbidden_hits" and new > old + max_forbidden_hit_increase:
                item["regression"] = True
                blockers.append(f"{metric}: increased by {new - old}, exceeding allowed increase {max_forbidden_hit_increase}")
            else:
                item["regression"] = False
        else:
            item["delta"] = None
            item["regression"] = None
            if old is not None or new is not None:
                warnings.append(f"Cannot compare {stage_name}.{metric}: numeric metric missing from baseline or candidate.")
            if stage_name == "behavior_bench" and metric in {"expected_pass_rate", "cases_with_forbidden_hits"}:
                blockers.append(f"{metric}: required regression metric is missing or invalid; refusing to pass the comparison gate.")
        comparisons.append(item)

    baseline_id = baseline.get("workflow_id")
    candidate_id = candidate.get("workflow_id")
    if baseline_id == candidate_id:
        warnings.append("Baseline and candidate share a workflow_id; verify that distinct runs were intended.")
    if baseline.get("config") != candidate.get("config"):
        warnings.append("Workflow configurations differ; metric changes may be confounded by settings.")
    baseline_case_hash = baseline.get("reproducibility", {}).get("case_suite", {}).get("sha256")
    candidate_case_hash = candidate.get("reproducibility", {}).get("case_suite", {}).get("sha256")
    if baseline_case_hash and candidate_case_hash and baseline_case_hash != candidate_case_hash:
        warnings.append("Case-suite SHA-256 fingerprints differ; results are not a strict same-suite comparison.")
    if baseline.get("model_id") != candidate.get("model_id"):
        warnings.append("Model IDs differ; this is a model comparison, not a controlled same-model regression check.")
    if baseline.get("quality_gate", {}).get("passed") is not True:
        warnings.append("Baseline workflow quality gate did not pass; interpret regression comparisons cautiously.")
    if candidate.get("quality_gate", {}).get("passed") is not True:
        blockers.append("candidate-workflow-quality-gate: candidate workflow did not pass its own gate")

    return {
        "schema_version": SCHEMA_VERSION,
        "comparison_id": str(uuid.uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": {"workflow_id": baseline_id, "model_id": baseline.get("model_id")},
        "candidate": {"workflow_id": candidate_id, "model_id": candidate.get("model_id")},
        "thresholds": {
            "max_pass_rate_drop": max_pass_rate_drop,
            "max_forbidden_hit_increase": max_forbidden_hit_increase,
        },
        "gate": {"passed": not blockers, "blockers": blockers},
        "comparisons": comparisons,
        "warnings": warnings,
        "interpretation_limits": [
            "A passing comparison is not a general correctness or safety certification.",
            "Safety heuristic counts are triage indicators and do not affect the gate.",
            "Response-length deltas are descriptive and do not indicate answer quality.",
            "Comparisons are meaningful only when the underlying case suites and metric definitions are comparable.",
        ],
    }



def render_comparison_html(result: dict[str, Any]) -> str:
    """Render a self-contained, escaped HTML summary of a comparison result."""
    esc = lambda value: html.escape(str(value), quote=True)
    gate = result.get("gate", {})
    passed = gate.get("passed") is True
    status = "PASS" if passed else "BLOCKED"
    status_class = "pass" if passed else "blocked"

    def display(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    rows = []
    for item in result.get("comparisons", []):
        regression = item.get("regression")
        if regression is True:
            outcome = "Regression"
        elif regression is False:
            outcome = "No configured regression"
        else:
            outcome = "Not comparable"
        rows.append(
            "<tr>"
            f"<th scope=\"row\">{esc(item.get('label', item.get('metric', 'Metric')))}</th>"
            f"<td>{esc(display(item.get('baseline')))}</td>"
            f"<td>{esc(display(item.get('candidate')))}</td>"
            f"<td>{esc(display(item.get('delta')))}</td>"
            f"<td>{esc(outcome)}</td>"
            "</tr>"
        )
    if not rows:
        rows.append('<tr><td colspan="5">No metric comparisons were provided.</td></tr>')

    def list_items(values: Any, empty: str) -> str:
        if not values:
            return f"<li>{esc(empty)}</li>"
        return "".join(f"<li>{esc(value)}</li>" for value in values)

    baseline = result.get("baseline", {})
    candidate = result.get("candidate", {})
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NeuronLab workflow comparison</title>
<style>
:root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
body {{ max-width: 1100px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }}
.status {{ display: inline-block; padding: .25rem .65rem; border-radius: .4rem; font-weight: 700; }}
.pass {{ background: #d1fae5; color: #065f46; }}
.blocked {{ background: #fee2e2; color: #991b1b; }}
.meta {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; }}
.panel {{ border: 1px solid #8886; border-radius: .6rem; padding: 1rem; margin: 1rem 0; overflow-wrap: anywhere; }}
table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
th, td {{ padding: .65rem; border-bottom: 1px solid #8886; text-align: left; }}
th {{ vertical-align: top; }}
.note {{ color: #777; font-size: .95rem; }}
</style>
</head>
<body>
<h1>NeuronLab workflow comparison</h1>
<p><span class="status {status_class}">{status}</span> <span class="note">Comparison ID: {esc(result.get('comparison_id', 'unknown'))}</span></p>
<p>Created: {esc(result.get('created_at_utc', 'unknown'))}</p>
<div class="meta">
<section class="panel"><h2>Baseline</h2><p>Workflow: {esc(baseline.get('workflow_id'))}</p><p>Model: {esc(baseline.get('model_id'))}</p></section>
<section class="panel"><h2>Candidate</h2><p>Workflow: {esc(candidate.get('workflow_id'))}</p><p>Model: {esc(candidate.get('model_id'))}</p></section>
</div>
<h2>Metric comparison</h2>
<p class="note">Only the explicitly configured Behavior Bench metrics below affect the regression gate. Other metrics are diagnostics.</p>
<table>
<thead><tr><th scope="col">Metric</th><th scope="col">Baseline</th><th scope="col">Candidate</th><th scope="col">Delta</th><th scope="col">Assessment</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
<section class="panel"><h2>Gate blockers</h2><ul>{list_items(gate.get('blockers', []), 'No blockers reported.')}</ul></section>
<section class="panel"><h2>Warnings</h2><ul>{list_items(result.get('warnings', []), 'No warnings reported.')}</ul></section>
<section class="panel"><h2>Interpretation limits</h2><ul>{list_items(result.get('interpretation_limits', []), 'No interpretation limits provided.')}</ul></section>
<p class="note">This report is a regression aid, not a certification of model correctness or safety. Review the underlying workflow artifacts before making release decisions.</p>
</body>
</html>
"""



def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Earlier workflow.json")
    parser.add_argument("--candidate", type=Path, required=True, help="New workflow.json")
    parser.add_argument("--output", type=Path, default=Path("workflow-comparison.json"))
    parser.add_argument("--html-output", type=Path, default=None, help="HTML report path (defaults to the JSON output path with .html suffix)")
    parser.add_argument("--max-pass-rate-drop", type=float, default=0.05)
    parser.add_argument("--max-forbidden-hit-increase", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        result = compare_workflows(
            _load(args.baseline), _load(args.candidate),
            max_pass_rate_drop=args.max_pass_rate_drop,
            max_forbidden_hit_increase=args.max_forbidden_hit_increase,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        html_output = args.html_output or args.output.with_suffix(".html")
        html_output.parent.mkdir(parents=True, exist_ok=True)
        html_output.write_text(render_comparison_html(result), encoding="utf-8")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"HTML report: {html_output}")
    return 0 if result["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
