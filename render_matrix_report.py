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
    md = render_markdown(payload)
    # Escape the complete generated Markdown as text; model names and hypotheses are untrusted.
    escaped = html.escape(md)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Experiment Scientist Report</title>
<style>
body {{ font: 16px/1.55 system-ui, sans-serif; max-width: 1100px; margin: 2rem auto; padding: 0 1rem; color: #182230; }}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #f5f7fa; border: 1px solid #dce2ea; border-radius: 8px; padding: 1.25rem; }}
</style>
</head>
<body><h1>Experiment Scientist Report</h1><p>Readable text export of the matrix summary. Raw observations remain in the JSON/CSV files.</p><pre>{escaped}</pre></body>
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
