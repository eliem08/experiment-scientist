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
