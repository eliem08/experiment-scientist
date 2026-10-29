import json
from pathlib import Path
from types import SimpleNamespace

from unified_evaluation import run_workflow


def _fixture_repositories(root):
    behavior = root / "behavior-bench"
    safety = root / "safety-probe-lab"
    experiment = root / "experiment-scientist"
    steering = root / "steering-studio"
    flight = root / "neural-flight-school"
    (behavior / "examples").mkdir(parents=True)
    for directory in (safety, experiment, steering, flight):
        directory.mkdir()
    for directory, script in (
        (behavior, "benchmark.py"),
        (safety, "probe_eval.py"),
        (experiment, "prompt_experiment.py"),
        (steering, "activation_steering.py"),
        (flight, "attention_lab.py"),
    ):
        (directory / script).write_text("# test fixture\n", encoding="utf-8")
    cases = behavior / "examples" / "cases.jsonl"
    cases.write_text('{"id":"case-1","prompt":"hello"}\n', encoding="utf-8")
    return behavior, safety, experiment, steering, flight, cases


def _runner(command, *, cwd, env, capture_output, text, timeout, check):
    script = Path(command[1]).name
    mapping = {
        "benchmark.py": ("behavior-bench", "results.json"),
        "probe_eval.py": ("safety-probe-lab", "probe-results.json"),
        "prompt_experiment.py": ("experiment-scientist", "experiment.json"),
        "activation_steering.py": ("steering-studio", "steering.json"),
        "attention_lab.py": ("neural-flight-school", "report.json"),
    }
    project, filename = mapping[script]
    output = Path(command[command.index("--output") + 1])
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"project": project, "run_id": project + "-run"}
    payload = {"manifest": manifest}
    if project in {"behavior-bench", "experiment-scientist", "steering-studio"}:
        payload["model_id"] = command[command.index("--model-id") + 1]
        manifest["model_id"] = payload["model_id"]
    if project == "behavior-bench":
        payload.update({"cases": [], "summary": {"total": 1, "expected_pass_rate": 1.0, "cases_with_forbidden_hits": 0}, "quality_gate": {"passed": True}})
    elif project == "safety-probe-lab":
        payload["results"] = [{"heuristic_signal": False}]
    elif project == "experiment-scientist":
        payload["results"] = [{"word_delta_treatment_minus_baseline": 0}]
    elif project == "steering-studio":
        payload.update({"layer": 1, "scale": 0.2, "baseline": "base", "steered": "changed"})
    else:
        payload.update({"layers": [{"layer": 0}], "token_count": 5})
    (output / filename).write_text(json.dumps(payload), encoding="utf-8")
    return SimpleNamespace(returncode=0, stdout="ok", stderr="")


def test_optional_adapters_run_and_link_artifacts(tmp_path):
    behavior, safety, experiment, steering, flight, cases = _fixture_repositories(tmp_path)
    report = run_workflow(
        behavior_dir=behavior,
        safety_dir=safety,
        experiment_dir=experiment,
        steering_dir=steering,
        flight_school_dir=flight,
        model_id="toy-model",
        cases=cases,
        output_root=tmp_path / "outputs",
        device="cpu",
        runner=_runner,
    )

    stages = {stage["name"]: stage for stage in report["stages"]}
    assert set(stages) == {
        "behavior-bench",
        "safety-probe-lab",
        "experiment-scientist",
        "steering-studio",
        "neural-flight-school",
    }
    assert report["status"] == "completed"
    assert report["quality_gate"]["passed"] is True
    assert stages["steering-studio"]["manifest_workflow_id"] == report["workflow_id"]
    assert stages["neural-flight-school"]["manifest_workflow_id"] == report["workflow_id"]
    assert stages["neural-flight-school"]["artifact_model_id"] is None
    assert stages["steering-studio"]["metrics"]["layer"] == 1
    assert stages["neural-flight-school"]["metrics"]["layers"] == 1
    assert len(report["artifacts"]) == 5

    flight_artifact = json.loads(
        (tmp_path / "outputs" / "neural-flight-school" / "report.json").read_text(encoding="utf-8")
    )
    assert flight_artifact["manifest"]["workflow_id"] == report["workflow_id"]
