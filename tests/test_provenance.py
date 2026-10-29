from provenance import build_manifest

def test_manifest_is_json_serializable_and_hashes_inputs():
    import json
    manifest = build_manifest("test-project", model_id="toy-model", device="cpu", seed=3, inputs={"prompt": "hello"})
    assert manifest["schema_version"] == "neuronlab.run.v1"
    assert len(manifest["run_id"]) == 36
    assert manifest["project"] == "test-project"
    assert manifest["device"] == "cpu"
    assert len(manifest["input_sha256"]["prompt"]) == 64
    assert manifest["created_at_utc"].endswith("+00:00")
    json.dumps(manifest)


def test_manifest_inherits_unified_workflow_context(monkeypatch):
    monkeypatch.setenv("NEURONLAB_WORKFLOW_ID", "workflow-test-123")
    monkeypatch.setenv("NEURONLAB_WORKFLOW_SCHEMA", "neuronlab.workflow.v1")
    manifest = build_manifest("test-project")
    assert manifest["workflow_id"] == "workflow-test-123"
    assert manifest["workflow_schema_version"] == "neuronlab.workflow.v1"


def test_standalone_manifest_has_no_workflow_context(monkeypatch):
    monkeypatch.delenv("NEURONLAB_WORKFLOW_ID", raising=False)
    monkeypatch.delenv("NEURONLAB_WORKFLOW_SCHEMA", raising=False)
    manifest = build_manifest("test-project")
    assert manifest["workflow_id"] is None
    assert manifest["workflow_schema_version"] is None
