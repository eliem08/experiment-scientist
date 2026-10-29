"""Run Behavior Bench, Safety Probe Lab, and Experiment Scientist as one workflow.

This orchestrator reuses each tool's existing CLI, keeps its result files separate,
and writes workflow.json linking the stages by one workflow ID.
"""
from __future__ import annotations

import argparse
import json
import os
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
    max_new_tokens: int, seed: int,
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
            ],
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


def run_workflow(
    *, behavior_dir: Path, safety_dir: Path, experiment_dir: Path,
    model_id: str, cases: Path, output_root: Path, device: str = "auto",
    max_new_tokens: int = 96, seed: int = 17, timeout_seconds: int = 7200,
    runner=subprocess.run,
) -> dict[str, Any]:
    """Execute all stages and produce a machine-readable unified report."""
    workflow_id = str(uuid.uuid4())
    started_at = _utc_now()
    output_root.mkdir(parents=True, exist_ok=True)
    specs = build_stage_specs(
        behavior_dir=behavior_dir, safety_dir=safety_dir, experiment_dir=experiment_dir,
        model_id=model_id, cases=cases, output_root=output_root, device=device,
        max_new_tokens=max_new_tokens, seed=seed,
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
            if completed.returncode != 0:
                stage["status"] = "failed"
                stage["error"] = f"Process exited with code {completed.returncode}."
            elif not spec["artifact"].is_file():
                stage["status"] = "failed"
                stage["error"] = "Process succeeded but the expected result artifact was not created."
            else:
                stage["status"] = "completed"
                stage.update(_read_artifact_summary(spec["artifact"]))
                if stage.get("artifact_readable") is False:
                    stage["status"] = "failed"
                    stage["error"] = "Result artifact exists but is not valid JSON."
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
    report = {
        "schema_version": SCHEMA_VERSION,
        "workflow_id": workflow_id,
        "created_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "model_id": model_id,
        "device": device,
        "config": {"max_new_tokens": max_new_tokens, "seed": seed, "timeout_seconds": timeout_seconds},
        "status": "completed" if all(stage["status"] == "completed" for stage in stages) else "failed",
        "quality_gate": {"passed": not gate_blockers, "blockers": gate_blockers},
        "stages": stages,
        "artifacts": [stage["artifact"] for stage in stages if Path(stage["artifact"]).is_file()],
    }
    report_path = output_root / "workflow.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
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
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["quality_gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
