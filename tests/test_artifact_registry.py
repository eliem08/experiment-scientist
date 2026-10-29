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
