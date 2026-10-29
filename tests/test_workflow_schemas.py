import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _schema(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_workflow_schema_describes_reproducibility_fingerprints():
    schema = _schema("schemas/neuronlab.workflow.v1.schema.json")
    assert schema["properties"]["schema_version"]["const"] == "neuronlab.workflow.v1"
    fingerprints = schema["properties"]["reproducibility"]
    assert {"python_version", "platform", "case_suite", "runner_scripts"} <= set(
        fingerprints["required"]
    )
    assert fingerprints["properties"]["case_suite"]["properties"]["sha256"]["pattern"] == "^[a-f0-9]{64}$"


def test_comparison_schema_has_unique_id_and_explicit_gate():
    schema = _schema("schemas/neuronlab.workflow-comparison.v1.schema.json")
    assert schema["properties"]["schema_version"]["const"] == "neuronlab.workflow-comparison.v1"
    assert "comparison_id" in schema["required"]
    assert schema["properties"]["gate"]["required"] == ["passed", "blockers"]
    assert schema["properties"]["thresholds"]["properties"]["max_pass_rate_drop"]["maximum"] == 1
