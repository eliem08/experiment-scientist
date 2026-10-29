"""Run Behavior Bench, Safety Probe Lab, and Experiment Scientist as one workflow.

This orchestrator reuses each tool's existing CLI, keeps its result files separate,
and writes workflow.json linking the stages by one workflow ID.
"""
from __future__ import annotations

import argparse
import html
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

SCHEMA_VERSION = "neuronlab.workflow.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_stage_specs(
    *, behavior_dir: Path, safety_dir: Path, experiment_dir: Path,
    model_id: str, cases: Path, output_root: Path, device: str,
    max_new_tokens: int, seed: int, minimum_failing_severity: str | None = None,
) -> list[dict[str, Any]]:
    """Build explicit subprocess specs for the three supported local evaluators."""
    return [
        {
            "name": "behavior-bench",
            "cwd": behavior_dir,
            "script": behavior_dir / "benchmark.py",
            "output": output_root / "behavior-bench",
            "command": [
                sys.executable, str(behavior_dir / "benchmark.py"),
                "--model-id", model_id, "--cases", str(cases),
                "--output", str(output_root / "behavior-bench"),
                "--device", device, "--max-new-tokens", str(max_new_tokens),
            ] + (["--minimum-failing-severity", minimum_failing_severity] if minimum_failing_severity else []),
            "artifact": output_root / "behavior-bench" / "results.json",
        },
        {
            "name": "safety-probe-lab",
            "cwd": safety_dir,
            "script": safety_dir / "probe_eval.py",
            "output": output_root / "safety-probe-lab",
            "command": [
                sys.executable, str(safety_dir / "probe_eval.py"),
                "--model-id", model_id, "--output", str(output_root / "safety-probe-lab"),
                "--device", device, "--max-new-tokens", str(max_new_tokens),
            ],
            "artifact": output_root / "safety-probe-lab" / "probe-results.json",
        },
        {
            "name": "experiment-scientist",
            "cwd": experiment_dir,
            "script": experiment_dir / "prompt_experiment.py",
            "output": output_root / "experiment-scientist",
            "command": [
                sys.executable, str(experiment_dir / "prompt_experiment.py"),
                "--model-id", model_id, "--output", str(output_root / "experiment-scientist"),
                "--device", device, "--max-new-tokens", str(max_new_tokens), "--seed", str(seed),
            ],
            "artifact": output_root / "experiment-scientist" / "experiment.json",
        },
    ]


def _read_artifact_summary(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {"artifact_readable": False, "artifact_error": str(exc)}
    manifest = payload.get("manifest") if isinstance(payload, dict) else None
    summary = payload.get("summary") if isinstance(payload, dict) else None
    summary = summary if isinstance(summary, dict) else {}
    result: dict[str, Any] = {
        "artifact_readable": isinstance(payload, dict),
        "run_id": manifest.get("run_id") if isinstance(manifest, dict) else None,
        "manifest_workflow_id": manifest.get("workflow_id") if isinstance(manifest, dict) else None,
        "artifact_model_id": payload.get("model_id") or (manifest.get("model_id") if isinstance(manifest, dict) else None),
        "artifact_device": payload.get("device") or (manifest.get("device") if isinstance(manifest, dict) else None),
    }
    if isinstance(payload, dict):
        project = str(manifest.get("project", "")) if isinstance(manifest, dict) else ""
        if project == "behavior-bench":
            gate = payload.get("quality_gate", summary.get("quality_gate"))
            result["metrics"] = {
                "cases": summary.get("total", len(payload.get("cases", []))),
                "expected_pass_rate": summary.get("expected_pass_rate"),
                "cases_with_forbidden_hits": summary.get("cases_with_forbidden_hits"),
                "quality_gate_passed": gate.get("passed") if isinstance(gate, dict) else None,
            }
        elif project == "safety-probe-lab":
            rows = payload.get("results", [])
            rows = [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
            result["metrics"] = {
                "probes": len(rows),
                "heuristic_signals": sum(row.get("heuristic_signal") is True for row in rows),
                "warning": payload.get("warning"),
            }
        elif project == "experiment-scientist":
            rows = payload.get("results", [])
            rows = [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
            deltas = [row["word_delta_treatment_minus_baseline"] for row in rows
                      if isinstance(row.get("word_delta_treatment_minus_baseline"), (int, float))]
            result["metrics"] = {
                "paired_prompts": len(rows),
                "mean_word_delta": sum(deltas) / len(deltas) if deltas else None,
                "bootstrap_word_delta": payload.get("bootstrap_word_delta"),
                "interpretation": "Descriptive response-length difference, not answer quality.",
            }
    return result



def render_workflow_html(report: dict[str, Any], path: Path) -> None:
    """Write a self-contained human-readable report without embedding model responses."""
    rows = []
    for stage in report.get("stages", []):
        metrics = html.escape(json.dumps(stage.get("metrics", {}), indent=2, ensure_ascii=False))
        cells = [
            stage.get("name", ""),
            stage.get("status", ""),
            stage.get("exit_code", ""),
            stage.get("duration_seconds", ""),
            stage.get("artifact_model_id", ""),
            stage.get("artifact_device", ""),
            stage.get("manifest_workflow_id", ""),
            stage.get("note", ""),
            stage.get("error", ""),
        ]
        rows.append(
            "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in cells)
            + f"<td><pre>{metrics}</pre></td></tr>"
        )
    reproducibility = report.get("reproducibility", {})
    case_suite = reproducibility.get("case_suite", {}) if isinstance(reproducibility, dict) else {}
    runner_scripts = reproducibility.get("runner_scripts", {}) if isinstance(reproducibility, dict) else {}
    fingerprint_rows = []
    if isinstance(case_suite, dict) and case_suite:
        fingerprint_rows.append(
            "<tr><td>case suite</td><td>" + html.escape(str(case_suite.get("path", "")))
            + "</td><td><code>" + html.escape(str(case_suite.get("sha256", ""))) + "</code></td></tr>"
        )
    if isinstance(runner_scripts, dict):
        for name, details in sorted(runner_scripts.items()):
            if isinstance(details, dict):
                fingerprint_rows.append(
                    "<tr><td>" + html.escape(str(name)) + "</td><td>" + html.escape(str(details.get("path", "")))
                    + "</td><td><code>" + html.escape(str(details.get("sha256", ""))) + "</code></td></tr>"
                )
    fingerprint_table = "\n".join(fingerprint_rows) or '<tr><td colspan="3">No fingerprints recorded.</td></tr>'
    gate = report.get("quality_gate", {})
    gate_status = "PASS" if gate.get("passed") is True else "FAIL"
    blockers = ", ".join(str(value) for value in gate.get("blockers", [])) or "None"
    stage_rows = "\n".join(rows) or '<tr><td colspan="10">No stages recorded.</td></tr>'
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NeuronLab Unified Evaluation</title>
<style>
body{{font:16px/1.5 system-ui,sans-serif;max-width:1400px;margin:2rem auto;padding:0 1rem;color:#172033}}
h1,h2{{line-height:1.2}} .summary{{display:flex;flex-wrap:wrap;gap:1rem;margin:1.5rem 0}}
.card{{border:1px solid #ccd3df;border-radius:10px;padding:1rem;min-width:180px}}
.status{{font-weight:700}} table{{width:100%;border-collapse:collapse;display:block;overflow-x:auto}}
th,td{{border:1px solid #d7dce5;padding:.65rem;text-align:left;vertical-align:top}}
th{{background:#f3f5f8}} pre{{white-space:pre-wrap;max-width:360px;margin:0;font-size:.85rem}}
small{{color:#4b5563}}
</style>
</head>
<body>
<h1>NeuronLab Unified Evaluation</h1>
<p><small>Workflow ID: {html.escape(str(report.get("workflow_id", "")))} | Model: {html.escape(str(report.get("model_id", "")))} | Device: {html.escape(str(report.get("device", "")))}</small></p>
<div class="summary">
<div class="card"><strong>Workflow status</strong><div>{html.escape(str(report.get("status", "")))}</div></div>
<div class="card"><strong>Quality gate</strong><div class="status">{gate_status}</div></div>
<div class="card"><strong>Stages</strong><div>{len(report.get("stages", []))}</div></div>
<div class="card"><strong>Blockers</strong><div>{html.escape(blockers)}</div></div>
</div>
<h2>Stage results</h2>
<table>
<thead><tr><th>Stage</th><th>Status</th><th>Exit code</th><th>Seconds</th><th>Model ID</th><th>Resolved device</th><th>Workflow ID</th><th>Note</th><th>Error</th><th>Metrics</th></tr></thead>
<tbody>{stage_rows}</tbody>
</table>
<h2>Reproducibility fingerprints</h2>
<p><small>Python: {html.escape(str(reproducibility.get("python_version", "unknown")))} | Platform: {html.escape(str(reproducibility.get("platform", "unknown")))}</small></p>
<table><thead><tr><th>Input</th><th>Path</th><th>SHA-256</th></tr></thead><tbody>{fingerprint_table}</tbody></table>
<h2>Interpretation limits</h2>
<p>Behavior Bench checks are configured regression signals, not a general semantic-quality guarantee. Safety Probe Lab marker matches are heuristic triage indicators, not a validated safety score. Experiment Scientist word deltas describe response length, not answer quality. A passing workflow is not a safety or correctness certification.</p>
<p><small>Created: {html.escape(str(report.get("created_at_utc", "")))} | Completed: {html.escape(str(report.get("completed_at_utc", "")))}</small></p>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")

def run_workflow(
    *, behavior_dir: Path, safety_dir: Path, experiment_dir: Path,
    model_id: str, cases: Path, output_root: Path, device: str = "auto",
    max_new_tokens: int = 96, seed: int = 17, timeout_seconds: int = 7200,
    minimum_failing_severity: str | None = None, runner=subprocess.run,
) -> dict[str, Any]:
    """Execute all stages and produce a machine-readable unified report."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    if timeout_seconds < 1:
        raise ValueError("timeout_seconds must be at least 1")
    if not model_id.strip():
        raise ValueError("model_id must not be empty")
    workflow_id = str(uuid.uuid4())
    started_at = _utc_now()
    output_root.mkdir(parents=True, exist_ok=True)
    specs = build_stage_specs(
        behavior_dir=behavior_dir, safety_dir=safety_dir, experiment_dir=experiment_dir,
        model_id=model_id, cases=cases, output_root=output_root, device=device,
        max_new_tokens=max_new_tokens, seed=seed, minimum_failing_severity=minimum_failing_severity,
    )
    for spec in specs:
        if not spec["cwd"].is_dir():
            raise FileNotFoundError(f"{spec['name']} repository directory does not exist: {spec['cwd']}")
        if not spec["script"].is_file():
            raise FileNotFoundError(f"{spec['name']} runner does not exist: {spec['script']}")
    if not cases.is_file():
        raise FileNotFoundError(f"Behavior Bench case file does not exist: {cases}")

    stages: list[dict[str, Any]] = []
    for spec in specs:
        spec["output"].mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["NEURONLAB_WORKFLOW_ID"] = workflow_id
        env["NEURONLAB_WORKFLOW_SCHEMA"] = SCHEMA_VERSION
        log_path = spec["output"] / "workflow.log"
        started = time.monotonic()
        stage: dict[str, Any] = {
            "name": spec["name"],
            "status": "running",
            "command": spec["command"],
            "artifact": str(spec["artifact"]),
            "log": str(log_path),
        }
        try:
            completed = runner(
                spec["command"], cwd=str(spec["cwd"]), env=env,
                capture_output=True, text=True, timeout=timeout_seconds, check=False,
            )
            log_path.write_text(
                (completed.stdout or "") + ("\n--- STDERR ---\n" if completed.stderr else "") + (completed.stderr or ""),
                encoding="utf-8",
            )
            stage["exit_code"] = completed.returncode
            stage["duration_seconds"] = round(time.monotonic() - started, 3)
            if not spec["artifact"].is_file():
                stage["status"] = "failed"
                stage["error"] = f"Process exited with code {completed.returncode} and did not create the expected result artifact."
            else:
                stage["status"] = "completed"
                stage.update(_read_artifact_summary(spec["artifact"]))
                if stage.get("artifact_readable") is False:
                    stage["status"] = "failed"
                    stage["error"] = "Result artifact exists but is not valid JSON."
                elif stage.get("manifest_workflow_id") != workflow_id:
                    stage["status"] = "failed"
                    stage["error"] = "Result manifest workflow_id does not match this invocation; refusing to mix stale or unlinked results."
                elif stage.get("artifact_model_id") is not None and stage.get("artifact_model_id") != model_id:
                    stage["status"] = "failed"
                    stage["error"] = "Result model ID does not match the requested workflow model."
                elif completed.returncode != 0:
                    expected_gate_exit = (
                        spec["name"] == "behavior-bench"
                        and completed.returncode == 1
                        and stage.get("metrics", {}).get("quality_gate_passed") is False
                        and '"quality_gate"' in (completed.stdout or "")
                        and not (completed.stderr or "").strip()
                    )
                    if expected_gate_exit:
                        stage["note"] = "Behavior Bench returned exit code 1 because its configured regression gate failed; the artifact is validated and retained."
                    else:
                        stage["status"] = "failed"
                        stage["error"] = f"Process exited with code {completed.returncode}."
        except subprocess.TimeoutExpired:
            stage["status"] = "failed"
            stage["error"] = f"Stage exceeded timeout of {timeout_seconds} seconds."
            stage["duration_seconds"] = round(time.monotonic() - started, 3)
            log_path.write_text("Stage timed out. Check model download or inference progress.\n", encoding="utf-8")
        except OSError as exc:
            stage["status"] = "failed"
            stage["error"] = str(exc)
            stage["duration_seconds"] = round(time.monotonic() - started, 3)
            log_path.write_text(f"Could not start stage: {exc}\n", encoding="utf-8")
        stages.append(stage)

    behavior = next((stage for stage in stages if stage["name"] == "behavior-bench"), {})
    behavior_metrics = behavior.get("metrics", {})
    gate_blockers = [stage["name"] for stage in stages if stage["status"] != "completed"]
    if behavior_metrics.get("quality_gate_passed") is False:
        gate_blockers.append("behavior-bench-quality-gate")
    stage_metrics = {stage["name"]: stage.get("metrics", {}) for stage in stages}
    workflow_summary = {
        "stages_total": len(stages),
        "stages_completed": sum(stage["status"] == "completed" for stage in stages),
        "stages_failed": sum(stage["status"] == "failed" for stage in stages),
        "behavior_bench": stage_metrics.get("behavior-bench", {}),
        "safety_probe_lab": stage_metrics.get("safety-probe-lab", {}),
        "experiment_scientist": stage_metrics.get("experiment-scientist", {}),
    }
    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    report = {
        "schema_version": SCHEMA_VERSION,
        "workflow_id": workflow_id,
        "created_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "model_id": model_id,
        "device": device,
        "config": {"max_new_tokens": max_new_tokens, "seed": seed, "timeout_seconds": timeout_seconds, "minimum_failing_severity": minimum_failing_severity},
        "reproducibility": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "case_suite": {
                "path": str(cases.resolve()),
                "size_bytes": cases.stat().st_size,
                "sha256": sha256_file(cases),
            },
            "runner_scripts": {
                spec["name"]: {
                    "path": str(spec["script"].resolve()),
                    "sha256": sha256_file(spec["script"]),
                }
                for spec in specs
            },
        },
        "status": "completed" if all(stage["status"] == "completed" for stage in stages) else "failed",
        "quality_gate": {"passed": not gate_blockers, "blockers": gate_blockers},
        "summary": workflow_summary,
        "stages": stages,
        "artifacts": [stage["artifact"] for stage in stages if stage["status"] == "completed" and Path(stage["artifact"]).is_file()],
    }
    report_path = output_root / "workflow.json"
    html_report_path = output_root / "workflow-report.html"
    report["report_path"] = str(report_path)
    report["html_report_path"] = str(html_report_path)
    render_workflow_html(report, html_report_path)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavior-bench-dir", type=Path, default=here.parent / "behavior-bench")
    parser.add_argument("--safety-probe-dir", type=Path, default=here.parent / "safety-probe-lab")
    parser.add_argument("--experiment-scientist-dir", type=Path, default=here)
    parser.add_argument("--model-id", default="distilgpt2")
    parser.add_argument("--cases", type=Path, help="JSONL case suite; defaults to behavior-bench/examples/cases.jsonl")
    parser.add_argument("--output", type=Path, default=Path("outputs/unified-evaluation"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--minimum-failing-severity", choices=("low", "normal", "high", "critical"))
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args(argv)
    behavior_dir = args.behavior_bench_dir.expanduser().resolve()
    safety_dir = args.safety_probe_dir.expanduser().resolve()
    experiment_dir = args.experiment_scientist_dir.expanduser().resolve()
    cases = (args.cases or (behavior_dir / "examples" / "cases.jsonl")).expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    try:
        report = run_workflow(
            behavior_dir=behavior_dir, safety_dir=safety_dir, experiment_dir=experiment_dir,
            model_id=args.model_id, cases=cases, output_root=output_root, device=args.device,
            max_new_tokens=args.max_new_tokens, seed=args.seed, timeout_seconds=args.timeout_seconds,
            minimum_failing_severity=args.minimum_failing_severity,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["quality_gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
