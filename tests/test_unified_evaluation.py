import json
from pathlib import Path
from types import SimpleNamespace

from unified_evaluation import run_workflow


def _make_repositories(tmp_path):
    behavior = tmp_path / "behavior-bench"
    safety = tmp_path / "safety-probe-lab"
    experiment = tmp_path / "experiment-scientist"
    (behavior / "examples").mkdir(parents=True)
    safety.mkdir()
    experiment.mkdir()
    (behavior / "benchmark.py").write_text("# test fixture\n", encoding="utf-8")
    (safety / "probe_eval.py").write_text("# test fixture\n", encoding="utf-8")
    (experiment / "prompt_experiment.py").write_text("# test fixture\n", encoding="utf-8")
    cases = behavior / "examples" / "cases.jsonl"
    cases.write_text('{"id":"case-1","prompt":"hello"}\n', encoding="utf-8")
    return behavior, safety, experiment, cases


def _fake_runner(*, behavior_gate=True, fail_stage=None, workflow_id_override=None, model_id_override=None, nonzero_on_behavior_gate=False):
    def run(command, *, cwd, env, capture_output, text, timeout, check):
        script = Path(command[1]).name
        stage_by_script = {
            "benchmark.py": ("behavior-bench", "results.json"),
            "probe_eval.py": ("safety-probe-lab", "probe-results.json"),
            "prompt_experiment.py": ("experiment-scientist", "experiment.json"),
        }
        project, filename = stage_by_script[script]
        if project == fail_stage:
            return SimpleNamespace(returncode=2, stdout="", stderr="fixture failure")
        output = Path(command[command.index("--output") + 1])
        output.mkdir(parents=True, exist_ok=True)
        requested_model = command[command.index("--model-id") + 1]
        artifact_model = model_id_override or requested_model
        payload = {
            "model_id": artifact_model,
            "manifest": {
                "project": project,
                "model_id": artifact_model,
                "device": "cpu",
                "run_id": f"{project}-run",
                "workflow_id": workflow_id_override or env["NEURONLAB_WORKFLOW_ID"],
                "workflow_schema_version": env["NEURONLAB_WORKFLOW_SCHEMA"],
            }
        }
        if project == "behavior-bench":
            payload.update({
                "cases": [{"id": "case-1"}],
                "summary": {"total": 1, "expected_pass_rate": 1.0, "cases_with_forbidden_hits": 0},
                "quality_gate": {"passed": behavior_gate},
            })
        elif project == "safety-probe-lab":
            payload["results"] = [{"heuristic_signal": True}, {"heuristic_signal": False}]
        else:
            payload.update({
                "results": [{"word_delta_treatment_minus_baseline": 2}],
                "bootstrap_word_delta": {"mean": 2, "lower": None, "upper": None},
            })
        (output / filename).write_text(json.dumps(payload), encoding="utf-8")
        if project == "behavior-bench" and not behavior_gate and nonzero_on_behavior_gate:
            return SimpleNamespace(returncode=1, stdout=json.dumps({"quality_gate": {"passed": False}}), stderr="")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")
    return run


def test_unified_workflow_runs_all_stages_with_shared_workflow_id(tmp_path):
    behavior, safety, experiment, cases = _make_repositories(tmp_path)
    output = tmp_path / "outputs"
    report = run_workflow(
        behavior_dir=behavior, safety_dir=safety, experiment_dir=experiment,
        model_id="toy-model", cases=cases, output_root=output, device="cpu",
        runner=_fake_runner(),
    )
    assert report["schema_version"] == "neuronlab.workflow.v1"
    assert report["status"] == "completed"
    assert report["quality_gate"]["passed"] is True
    assert len(report["stages"]) == 3
    assert all(stage["status"] == "completed" for stage in report["stages"])
    assert all(stage["manifest_workflow_id"] == report["workflow_id"] for stage in report["stages"])
    assert all(stage["artifact_device"] == "cpu" for stage in report["stages"])
    saved = json.loads((output / "workflow.json").read_text(encoding="utf-8"))
    assert saved["workflow_id"] == report["workflow_id"]
    assert saved["report_path"] == str(output / "workflow.json")
    assert saved["html_report_path"] == str(output / "workflow-report.html")
    html_report = (output / "workflow-report.html").read_text(encoding="utf-8")
    assert "NeuronLab Unified Evaluation" in html_report
    assert "PASS" in html_report
    assert "Reproducibility fingerprints" in html_report
    assert "SHA-256" in html_report
    assert len(saved["artifacts"]) == 3
    assert saved["summary"]["stages_completed"] == 3
    assert saved["summary"]["behavior_bench"]["expected_pass_rate"] == 1.0
    assert saved["summary"]["safety_probe_lab"]["heuristic_signals"] == 1
    assert saved["summary"]["experiment_scientist"]["mean_word_delta"] == 2.0
    fingerprints = saved["reproducibility"]
    assert fingerprints["python_version"]
    assert len(fingerprints["case_suite"]["sha256"]) == 64
    assert fingerprints["case_suite"]["size_bytes"] > 0
    assert set(fingerprints["runner_scripts"]) == {
        "behavior-bench", "safety-probe-lab", "experiment-scientist"
    }
    assert all(len(item["sha256"]) == 64 for item in fingerprints["runner_scripts"].values())


def test_unified_workflow_fails_gate_if_behavior_quality_gate_fails(tmp_path):
    behavior, safety, experiment, cases = _make_repositories(tmp_path)
    report = run_workflow(
        behavior_dir=behavior, safety_dir=safety, experiment_dir=experiment,
        model_id="toy-model", cases=cases, output_root=tmp_path / "outputs",
        runner=_fake_runner(behavior_gate=False),
    )
    assert report["status"] == "completed"
    assert report["quality_gate"]["passed"] is False
    assert "behavior-bench-quality-gate" in report["quality_gate"]["blockers"]


def test_unified_workflow_continues_and_records_failed_stage(tmp_path):
    behavior, safety, experiment, cases = _make_repositories(tmp_path)
    report = run_workflow(
        behavior_dir=behavior, safety_dir=safety, experiment_dir=experiment,
        model_id="toy-model", cases=cases, output_root=tmp_path / "outputs",
        runner=_fake_runner(fail_stage="safety-probe-lab"),
    )
    stages = {stage["name"]: stage for stage in report["stages"]}
    assert report["status"] == "failed"
    assert report["quality_gate"]["passed"] is False
    assert stages["safety-probe-lab"]["status"] == "failed"
    assert stages["safety-probe-lab"]["exit_code"] == 2
    assert stages["experiment-scientist"]["status"] == "completed"


def test_unified_workflow_passes_configured_severity_to_behavior_bench(tmp_path):
    from unified_evaluation import build_stage_specs

    behavior, safety, experiment, cases = _make_repositories(tmp_path)
    specs = build_stage_specs(
        behavior_dir=behavior, safety_dir=safety, experiment_dir=experiment,
        model_id="toy-model", cases=cases, output_root=tmp_path / "out",
        device="cpu", max_new_tokens=20, seed=7, minimum_failing_severity="high",
    )
    command = specs[0]["command"]
    assert command[command.index("--minimum-failing-severity") + 1] == "high"
    assert "--minimum-failing-severity" not in specs[1]["command"]
    assert "--minimum-failing-severity" not in specs[2]["command"]


def test_unified_workflow_rejects_stale_artifacts_with_wrong_workflow_id(tmp_path):
    behavior, safety, experiment, cases = _make_repositories(tmp_path)
    report = run_workflow(
        behavior_dir=behavior, safety_dir=safety, experiment_dir=experiment,
        model_id="toy-model", cases=cases, output_root=tmp_path / "outputs",
        runner=_fake_runner(workflow_id_override="stale-workflow"),
    )
    assert report["status"] == "failed"
    assert report["quality_gate"]["passed"] is False
    assert all(stage["status"] == "failed" for stage in report["stages"])
    assert all("does not match" in stage["error"] for stage in report["stages"])
    assert report["artifacts"] == []


def test_unified_workflow_rejects_mixed_model_artifacts(tmp_path):
    behavior, safety, experiment, cases = _make_repositories(tmp_path)
    report = run_workflow(
        behavior_dir=behavior, safety_dir=safety, experiment_dir=experiment,
        model_id="requested-model", cases=cases, output_root=tmp_path / "outputs",
        runner=_fake_runner(model_id_override="different-model"),
    )
    assert report["status"] == "failed"
    assert all(stage["status"] == "failed" for stage in report["stages"])
    assert all("model ID does not match" in stage["error"] for stage in report["stages"])
    assert report["artifacts"] == []


def test_unified_workflow_rejects_invalid_runtime_limits(tmp_path):
    import pytest

    behavior, safety, experiment, cases = _make_repositories(tmp_path)
    common = dict(
        behavior_dir=behavior, safety_dir=safety, experiment_dir=experiment,
        model_id="toy-model", cases=cases, output_root=tmp_path / "outputs",
        runner=_fake_runner(),
    )
    with pytest.raises(ValueError, match="max_new_tokens"):
        run_workflow(**common, max_new_tokens=0)
    with pytest.raises(ValueError, match="timeout_seconds"):
        run_workflow(**common, timeout_seconds=0)


def test_html_report_escapes_untrusted_stage_fields(tmp_path):
    from unified_evaluation import render_workflow_html

    path = tmp_path / "report.html"
    render_workflow_html({
        "workflow_id": "<script>alert(1)</script>",
        "model_id": "toy-model",
        "device": "cpu",
        "status": "failed",
        "quality_gate": {"passed": False, "blockers": ["unsafe <stage>"]},
        "stages": [{"name": "<img src=x>", "status": "failed", "error": "<script>"}],
    }, path)
    rendered = path.read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in rendered
    assert "<img src=x>" not in rendered
    assert "&lt;script&gt;" in rendered


def test_expected_behavior_gate_exit_retains_valid_artifact_without_marking_stage_crashed(tmp_path):
    behavior, safety, experiment, cases = _make_repositories(tmp_path)
    output = tmp_path / "outputs"
    report = run_workflow(
        behavior_dir=behavior, safety_dir=safety, experiment_dir=experiment,
        model_id="toy-model", cases=cases, output_root=output,
        minimum_failing_severity="high",
        runner=_fake_runner(behavior_gate=False, nonzero_on_behavior_gate=True),
    )
    behavior_stage = next(stage for stage in report["stages"] if stage["name"] == "behavior-bench")
    assert behavior_stage["status"] == "completed"
    assert behavior_stage["exit_code"] == 1
    assert "regression gate failed" in behavior_stage["note"]
    assert report["status"] == "completed"
    assert report["quality_gate"]["passed"] is False
    assert len(report["artifacts"]) == 3
