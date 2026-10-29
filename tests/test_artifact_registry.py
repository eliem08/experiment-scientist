import json
from pathlib import Path

from artifact_registry import CONTRACT_VERSION, inspect_artifact, scan_artifacts


def _write(path: Path, *, run_id="run-1", manifest_version=CONTRACT_VERSION):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "model_id": "toy-model",
        "manifest": {"schema_version": manifest_version, "run_id": run_id,
                     "project": "behavior-bench", "created_at_utc": "2026-10-29T10:00:00+00:00",
                     "model_id": "toy-model"},
        "cases": [{"id": "a"}, {"id": "b"}],
    }), encoding="utf-8")
    return path


def test_inspect_artifact_normalizes_shared_manifest(tmp_path):
    item = inspect_artifact(_write(tmp_path / "results.json"))
    assert item["manifest_status"] == "valid"
    assert item["schema_version"] == CONTRACT_VERSION
    assert item["run_id"] == "run-1"
    assert item["row_count"] == 2
    assert item["model_id"] == "toy-model"


def test_scan_artifacts_finds_results_and_reports_legacy_manifest(tmp_path):
    _write(tmp_path / "one" / "results.json")
    _write(tmp_path / "two" / "probe-results.json", run_id="run-2", manifest_version="old")
    (tmp_path / "ignore.json").write_text("{}", encoding="utf-8")
    result = scan_artifacts([tmp_path])
    assert result["summary"]["artifacts"] == 2
    assert result["summary"]["valid_manifests"] == 1
    assert result["summary"]["legacy_manifests"] == 1


def test_scan_artifacts_detects_duplicate_run_ids_and_bad_files(tmp_path):
    _write(tmp_path / "a" / "results.json")
    _write(tmp_path / "b" / "results.json")
    bad = tmp_path / "c" / "experiment.json"
    bad.parent.mkdir()
    bad.write_text("{broken", encoding="utf-8")
    result = scan_artifacts([tmp_path])
    assert result["summary"]["duplicate_run_ids"] == 1
    assert result["duplicate_run_ids"][0]["run_id"] == "run-1"
    assert result["summary"]["read_errors"] == 1


def test_scan_reports_missing_root_and_deduplicates_same_path(tmp_path):
    path = _write(tmp_path / "results.json")
    result = scan_artifacts([tmp_path, path, tmp_path / "missing"])
    assert result["summary"]["artifacts"] == 1
    assert result["summary"]["read_errors"] == 1


def test_extract_numeric_metrics_reads_nested_scalars_and_skips_manifest():
    from artifact_registry import extract_numeric_metrics

    metrics = extract_numeric_metrics({
        "manifest": {"seed": 17, "schema_version": CONTRACT_VERSION},
        "summary": {"pass_rate": 0.75, "count": 4, "enabled": True},
        "model": {"latency": 12.5},
        "cases": [{"score": 1.0}],
    })
    assert metrics == {"summary.pass_rate": 0.75, "summary.count": 4, "model.latency": 12.5}


def test_extract_numeric_metrics_is_bounded():
    from artifact_registry import extract_numeric_metrics

    metrics = extract_numeric_metrics({f"metric_{i}": i for i in range(20)}, max_metrics=5)
    assert len(metrics) == 5


def test_cli_exports_portable_json_report(tmp_path, capsys):
    from workbench_cli import main

    _write(tmp_path / "results.json")
    code = main([str(tmp_path)])
    output = json.loads(capsys.readouterr().out)
    assert code == 0
    assert output["summary"]["artifacts"] == 1
    assert output["artifacts"][0]["run_id"] == "run-1"
    assert "payload" not in output["artifacts"][0]


def test_cli_writes_csv_and_returns_nonzero_on_read_errors(tmp_path, capsys):
    from workbench_cli import main

    _write(tmp_path / "results.json")
    bad = tmp_path / "broken" / "results.json"
    bad.parent.mkdir()
    bad.write_text("{bad json", encoding="utf-8")
    output_path = tmp_path / "history.csv"
    code = main([str(tmp_path), "--format", "csv", "--output", str(output_path)])
    assert code == 1
    content = output_path.read_text(encoding="utf-8")
    assert "project,model_id,run_id" in content
    assert "behavior-bench" in content
    assert capsys.readouterr().out == ""


def test_project_summaries_use_behavior_bench_quality_gate_and_metrics():
    from workbench_analytics import summarize_artifact

    artifact = {
        "project": "behavior-bench",
        "payload": {
            "manifest": {"project": "behavior-bench"},
            "quality_gate": {"passed": False},
            "summary": {"total": 4, "expected_pass_rate": 0.75, "cases_with_forbidden_hits": 1},
            "cases": [{"structured_pass": True}, {"structured_pass": False}],
        },
    }
    metrics = {row["metric"]: row["value"] for row in summarize_artifact(artifact)}
    assert metrics["Cases evaluated"] == 4
    assert metrics["Expected-check pass rate"] == 0.75
    assert metrics["Quality gate"] == "FAIL"
    assert metrics["Structured JSON pass rate"] == 0.5


def test_project_summaries_mark_safety_metrics_as_heuristics():
    from workbench_analytics import summarize_artifact

    artifact = {
        "project": "safety-probe-lab",
        "payload": {"results": [
            {"heuristic_signal": True, "marker_count": 2},
            {"heuristic_signal": False, "marker_count": 0},
        ]},
    }
    metrics = {row["metric"]: row["value"] for row in summarize_artifact(artifact)}
    assert metrics["Probes evaluated"] == 2
    assert metrics["Heuristic signal count"] == 1
    assert metrics["Heuristic signal rate"] == 0.5


def test_project_summaries_report_paired_experiment_delta_without_claiming_quality():
    from workbench_analytics import summarize_artifact

    artifact = {
        "project": "experiment-scientist",
        "payload": {
            "results": [
                {"word_delta_treatment_minus_baseline": 2},
                {"word_delta_treatment_minus_baseline": -1},
            ],
            "bootstrap_word_delta": {"mean": 0.5, "lower": -2.0, "upper": 3.0},
        },
    }
    metrics = {row["metric"]: row["value"] for row in summarize_artifact(artifact)}
    assert metrics["Paired prompts"] == 2
    assert metrics["Mean treatment-minus-baseline word delta"] == 0.5
    assert metrics["Bootstrap CI lower"] == -2.0


def test_compare_metric_maps_adds_deltas_against_first_run_only_for_common_metrics():
    from workbench_analytics import compare_metric_maps

    rows = compare_metric_maps(
        [{"pass_rate": 0.8, "latency": 100}, {"pass_rate": 0.9, "latency": 120, "extra": 1}],
        ["baseline", "candidate"],
    )
    assert rows == [
        {"metric": "latency", "baseline": 100, "candidate": 120, "delta vs baseline (candidate)": 20},
        {"metric": "pass_rate", "baseline": 0.8, "candidate": 0.9, "delta vs baseline (candidate)": 0.09999999999999998},
    ]


def test_compare_metric_maps_requires_two_runs():
    from workbench_analytics import compare_metric_maps

    assert compare_metric_maps([{"x": 1}], ["only"]) == []


def test_registry_recognizes_unified_workflow_reports(tmp_path):
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps({
        "schema_version": "neuronlab.workflow.v1",
        "workflow_id": "workflow-123",
        "created_at_utc": "2026-10-29T10:00:00+00:00",
        "model_id": "toy-model",
        "status": "completed",
        "quality_gate": {"passed": True, "blockers": []},
        "stages": [{"name": "behavior-bench"}, {"name": "safety-probe-lab"}],
    }), encoding="utf-8")
    item = inspect_artifact(path)
    assert item["manifest_status"] == "valid"
    assert item["project"] == "neuronlab-unified-workflow"
    assert item["run_id"] == "workflow-123"
    assert item["workflow_id"] == "workflow-123"
    assert item["schema_version"] == "neuronlab.workflow.v1"
    assert item["row_count"] == 2


def test_project_summaries_surface_unified_workflow_gate_and_stage_failures():
    from workbench_analytics import summarize_artifact

    artifact = {
        "project": "neuronlab-unified-workflow",
        "payload": {
            "schema_version": "neuronlab.workflow.v1",
            "status": "failed",
            "stages": [{"status": "completed"}, {"status": "failed"}],
            "quality_gate": {"passed": False, "blockers": ["safety-probe-lab"]},
            "summary": {
                "stages_total": 3, "stages_completed": 2, "stages_failed": 1,
                "behavior_bench": {"expected_pass_rate": 0.8, "cases_with_forbidden_hits": 2},
                "safety_probe_lab": {"heuristic_signals": 3},
                "experiment_scientist": {"mean_word_delta": -1.5},
            },
        },
    }
    metrics = {row["metric"]: row["value"] for row in summarize_artifact(artifact)}
    assert metrics["Workflow status"] == "failed"
    assert metrics["Stages completed"] == 1
    assert metrics["Stages failed"] == 1
    assert metrics["Unified quality gate"] == "FAIL"
    assert metrics["Gate blockers"] == "safety-probe-lab"


def test_cli_includes_unified_workflow_status_and_gate_in_export(tmp_path, capsys):
    from workbench_cli import main

    path = tmp_path / "workflow.json"
    path.write_text(json.dumps({
        "schema_version": "neuronlab.workflow.v1",
        "workflow_id": "workflow-cli-test",
        "model_id": "toy-model",
        "status": "failed",
        "quality_gate": {"passed": False, "blockers": ["safety-probe-lab"]},
        "html_report_path": "/tmp/workflow-report.html",
        "stages": [],
    }), encoding="utf-8")
    assert main([str(path)]) == 0
    report = json.loads(capsys.readouterr().out)
    item = report["artifacts"][0]
    assert item["workflow_id"] == "workflow-cli-test"
    assert item["workflow_status"] == "failed"
    assert item["quality_gate_passed"] is False
    assert item["html_report_path"] == "/tmp/workflow-report.html"


def test_unified_workflow_scorecard_includes_cross_tool_metrics():
    from workbench_analytics import summarize_artifact

    artifact = {
        "project": "neuronlab-unified-workflow",
        "payload": {
            "schema_version": "neuronlab.workflow.v1",
            "status": "completed",
            "quality_gate": {"passed": True, "blockers": []},
            "stages": [{"status": "completed"}] * 3,
            "summary": {
                "stages_total": 3, "stages_completed": 3, "stages_failed": 0,
                "behavior_bench": {"expected_pass_rate": 0.9, "cases_with_forbidden_hits": 1},
                "safety_probe_lab": {"heuristic_signals": 2},
                "experiment_scientist": {"mean_word_delta": 4.5},
            },
        },
    }
    metrics = {row["metric"]: row["value"] for row in summarize_artifact(artifact)}
    assert metrics["Behavior expected-check pass rate"] == 0.9
    assert metrics["Behavior forbidden-hit cases"] == 1
    assert metrics["Safety heuristic signals"] == 2
    assert metrics["Experiment mean word delta"] == 4.5



def test_inspect_artifact_normalizes_workflow_comparison(tmp_path):
    path = tmp_path / "workflow-comparison.json"
    path.write_text(json.dumps({
        "schema_version": "neuronlab.workflow-comparison.v1",
        "comparison_id": "comparison-unique-1",
        "created_at_utc": "2026-10-29T10:00:00+00:00",
        "baseline": {"workflow_id": "old-run", "model_id": "toy-model"},
        "candidate": {"workflow_id": "new-run", "model_id": "toy-model"},
        "gate": {"passed": False, "blockers": ["expected_pass_rate regressed"]},
        "comparisons": [{"metric": "expected_pass_rate", "delta": -0.2}],
        "warnings": ["fixture warning"],
    }), encoding="utf-8")
    item = inspect_artifact(path)
    assert item["manifest_status"] == "valid"
    assert item["project"] == "neuronlab-workflow-comparison"
    assert item["workflow_id"] == "new-run"
    assert item["run_id"] == "comparison-unique-1"
    assert item["row_count"] == 1
    assert "fixture warning" in item["warnings"]


def test_scan_artifacts_discovers_workflow_comparison(tmp_path):
    path = tmp_path / "workflow-comparison.json"
    path.write_text(json.dumps({
        "schema_version": "neuronlab.workflow-comparison.v1",
        "comparison_id": "comparison-unique-2",
        "candidate": {"workflow_id": "new-run", "model_id": "toy-model"},
        "gate": {"passed": True, "blockers": []},
        "comparisons": [],
    }), encoding="utf-8")
    result = scan_artifacts([tmp_path])
    assert result["summary"]["artifacts"] == 1
    assert result["artifacts"][0]["project"] == "neuronlab-workflow-comparison"



def test_workbench_summarizes_workflow_comparison_gate_and_deltas():
    from workbench_analytics import summarize_artifact

    rows = summarize_artifact({
        "project": "neuronlab-workflow-comparison",
        "payload": {
            "schema_version": "neuronlab.workflow-comparison.v1",
            "baseline": {"model_id": "model-a"},
            "candidate": {"model_id": "model-b"},
            "gate": {"passed": False, "blockers": ["expected_pass_rate regressed"]},
            "comparisons": [
                {"label": "Behavior pass rate", "stage": "behavior-bench", "direction": "higher", "delta": -0.2},
                {"label": "Response length", "stage": "experiment-scientist", "direction": "neutral", "delta": 4.0},
            ],
        },
    })
    values = {row["metric"]: row["value"] for row in rows}
    assert values["Comparison gate"] == "FAIL"
    assert values["Baseline model"] == "model-a"
    assert values["Behavior pass rate delta"] == -0.2
    assert values["Response length delta"] == 4.0
    assert "Regression blockers" in values
