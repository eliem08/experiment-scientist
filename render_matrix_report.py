"""Render human-readable Markdown and HTML reports from matrix-results.json."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(payload: dict[str, Any]) -> str:
    experiment = payload.get("experiment", {})
    lines = [
        f"# Experiment report: {experiment.get('title', 'Untitled')}",
        "",
        f"**Hypothesis:** {experiment.get('hypothesis', 'Not recorded')}",
        "",
        f"- Primary metric: {experiment.get('primary_metric', 'not recorded')}",
        f"- Spec SHA-256: {experiment.get('spec_sha256', 'not recorded')}",
        f"- Models: {', '.join(payload.get('models', []))}",
        f"- Device: {payload.get('device', 'unknown')}",
        f"- Seeds: {', '.join(str(seed) for seed in payload.get('seeds', []))}",
        f"- Runtime: {_fmt(payload.get('runtime_seconds'))} seconds",
        "",
        "## Outcome summaries",
        "",
        "| Model | Metric | Direction | Cases | Seeds/case | Mean paired delta | Bootstrap interval | Cohen's dz |",
        "|---|---|---|---:|---:|---:|---|---:|",
    ]
    for row in payload.get("summaries", []):
        interval = f"[{_fmt(row.get('lower'))}, {_fmt(row.get('upper'))}]"
        lines.append(
            f"| {row.get('model_id', '')} | {row.get('metric_id', '')} | {row.get('direction', '')} "
            f"| {row.get('cases', '')} | {row.get('seeds_per_case', '')} "
            f"| {_fmt(row.get('mean'))} | {interval} | {_fmt(row.get('cohens_dz'))} |"
        )
    lines.extend([
        "",
        "## Performance",
        "",
        "| Model | Pairs | Baseline seconds | Treatment seconds | Baseline output tok/s | Treatment output tok/s |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in payload.get("performance_summary", []):
        lines.append(
            f"| {row.get('model_id', '')} | {row.get('paired_cases', '')} "
            f"| {_fmt(row.get('baseline_mean_generation_seconds'))} "
            f"| {_fmt(row.get('treatment_mean_generation_seconds'))} "
            f"| {_fmt(row.get('baseline_mean_output_tokens_per_second'))} "
            f"| {_fmt(row.get('treatment_mean_output_tokens_per_second'))} |"
        )
    lines.extend(["", "## Reproducibility", ""])
    repro = payload.get("reproducibility", {})
    for key in ("python_version", "platform", "torch_version", "transformers_version", "cuda_version", "github_sha", "run_config_sha256"):
        if repro.get(key) is not None:
            lines.append(f"- {key.replace('_', ' ').title()}: {repro[key]}")
    revisions = payload.get("model_revisions", {})
    for model, details in revisions.items():
        lines.append(f"- {model} model commit: {details.get('resolved_commit') or 'not resolved'}; tokenizer commit: {details.get('tokenizer_resolved_commit') or 'not resolved'}")
    lines.extend(["", "## Interpretation limits", ""])
    for item in payload.get("interpretation_limits", []):
        lines.append(f"- {item}")
    lines.extend([
        "",
        "Performance is measured on the current runtime/device; it is not a cloud price estimate. Outcome metrics only measure their declared rule and are not semantic-quality or safety judgments.",
        "",
    ])
    return "\n".join(lines)


def render_html(payload: dict[str, Any]) -> str:
    experiment = payload.get("experiment", {})
    esc = lambda value: html.escape(str(value if value is not None else "n/a"))
    metric_rows = []
    for row in payload.get("summaries", []):
        metric_rows.append(
            "<tr>" + "".join(f"<td>{esc(value)}</td>" for value in (
                row.get("model_id"), row.get("metric_id"), row.get("direction"),
                row.get("cases"), row.get("seeds_per_case"), _fmt(row.get("mean")),
                f"[{_fmt(row.get('lower'))}, {_fmt(row.get('upper'))}]",
                _fmt(row.get("cohens_dz")),
            )) + "</tr>"
        )
    performance_rows = []
    for row in payload.get("performance_summary", []):
        performance_rows.append(
            "<tr>" + "".join(f"<td>{esc(value)}</td>" for value in (
                row.get("model_id"), row.get("paired_cases"),
                _fmt(row.get("baseline_mean_generation_seconds")),
                _fmt(row.get("treatment_mean_generation_seconds")),
                _fmt(row.get("baseline_mean_output_tokens_per_second")),
                _fmt(row.get("treatment_mean_output_tokens_per_second")),
            )) + "</tr>"
        )
    repro = payload.get("reproducibility", {})
    repro_items = "".join(
        f"<li><strong>{esc(key.replace('_', ' ').title())}:</strong> {esc(repro[key])}</li>"
        for key in ("python_version", "platform", "torch_version", "transformers_version",
                    "cuda_version", "github_sha", "run_config_sha256")
        if repro.get(key) is not None
    )
    revisions = payload.get("model_revisions", {})
    for model, details in revisions.items():
        repro_items += f"<li><strong>{esc(model)} revisions:</strong> model {esc(details.get('resolved_commit') or 'not resolved')}; tokenizer {esc(details.get('tokenizer_resolved_commit') or 'not resolved')}</li>"
    limits = "".join(f"<li>{esc(item)}</li>" for item in payload.get("interpretation_limits", []))
    metric_rows_html = "".join(metric_rows) or '<tr><td colspan="8">No metric summaries available.</td></tr>'
    performance_rows_html = "".join(performance_rows) or '<tr><td colspan="6">No performance summaries available.</td></tr>'
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(experiment.get('title', 'Experiment Scientist Report'))}</title>
<style>
body {{ font: 16px/1.55 system-ui, sans-serif; max-width: 1200px; margin: 2rem auto; padding: 0 1rem; color: #182230; }}
h1, h2 {{ line-height: 1.2; }} .muted {{ color: #596579; overflow-wrap: anywhere; }}
table {{ width: 100%; border-collapse: collapse; display: block; overflow-x: auto; margin: 1rem 0 2rem; }}
th, td {{ border: 1px solid #dce2ea; padding: .65rem; text-align: left; vertical-align: top; }}
th {{ background: #f5f7fa; }} .panel {{ border: 1px solid #dce2ea; border-radius: 10px; padding: 1rem; margin: 1rem 0; }}
code {{ overflow-wrap: anywhere; }}
</style>
</head>
<body>
<h1>{esc(experiment.get('title', 'Experiment report'))}</h1>
<div class="panel"><strong>Hypothesis</strong><p>{esc(experiment.get('hypothesis', 'Not recorded'))}</p>
<p class="muted">Primary metric: {esc(experiment.get('primary_metric', 'not recorded'))} · Spec SHA-256: <code>{esc(experiment.get('spec_sha256', 'not recorded'))}</code></p>
<p class="muted">Models: {esc(', '.join(payload.get('models', [])))} · Device: {esc(payload.get('device', 'unknown'))} · Seeds: {esc(', '.join(str(seed) for seed in payload.get('seeds', [])))} · Runtime: {esc(_fmt(payload.get('runtime_seconds')))} seconds</p></div>
<h2>Outcome summaries</h2>
<table><thead><tr><th>Model</th><th>Metric</th><th>Direction</th><th>Cases</th><th>Seeds/case</th><th>Mean paired delta</th><th>Bootstrap interval</th><th>Cohen's dz</th></tr></thead><tbody>{metric_rows_html}</tbody></table>
<h2>Performance</h2>
<table><thead><tr><th>Model</th><th>Pairs</th><th>Baseline seconds</th><th>Treatment seconds</th><th>Baseline output tok/s</th><th>Treatment output tok/s</th></tr></thead><tbody>{performance_rows_html}</tbody></table>
<h2>Reproducibility</h2><ul>{repro_items or '<li>No runtime metadata recorded.</li>'}</ul>
<h2>Interpretation limits</h2><ul>{limits or '<li>No interpretation limits recorded.</li>'}</ul>
<p class="muted">Performance depends on runtime and hardware; it is not a hosted API price estimate. Rule-based metrics are not semantic-quality or safety judgments. Raw observations remain in the JSON/CSV files.</p>
</body>
</html>
"""

def load_result(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read matrix results {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != "experiment-scientist.matrix.v1":
        raise ValueError("Input must be an experiment-scientist.matrix.v1 result object")
    for key in ("experiment", "summaries", "observations"):
        if key not in payload:
            raise ValueError(f"Matrix results are missing required field {key!r}")
    if not isinstance(payload["experiment"], dict) or not isinstance(payload["summaries"], list) or not isinstance(payload["observations"], list):
        raise ValueError("Matrix results contain invalid experiment/summaries/observations types")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-html", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()
    try:
        payload = load_result(args.input)
        md_path = args.output_md or args.input.with_suffix(".md")
        html_path = args.output_html or args.input.with_suffix(".html")
        md_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_markdown(payload), encoding="utf-8")
        html_path.write_text(render_html(payload), encoding="utf-8")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"markdown": str(md_path), "html": str(html_path)}, indent=2))


if __name__ == "__main__":
    main()
