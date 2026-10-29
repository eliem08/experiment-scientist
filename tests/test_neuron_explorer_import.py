import json

from artifact_registry import scan_artifacts


def test_imports_precomputed_neuron_explorer_depth_report(tmp_path):
    report = tmp_path / "depth_report.json"
    report.write_text(json.dumps([
        {"layer": 3, "n_neurons": 384, "above_p95": 0.12},
        {"layer": 7, "n_neurons": 384, "above_p95": 0.18},
    ]), encoding="utf-8")

    result = scan_artifacts([tmp_path])

    assert result["summary"]["artifacts"] == 1
    artifact = result["artifacts"][0]
    assert artifact["project"] == "neuron-explorer"
    assert artifact["row_count"] == 2
    assert artifact["manifest_status"] == "legacy"
    assert "Missing manifest" in artifact["warnings"][0]
    assert artifact["payload"]["warning"].startswith("Imported precomputed Neuron Explorer")
