"""Read-only registry for NeuronLab run artifacts.

The registry normalizes provenance across the six standalone tools without
assuming their result-row schemas are identical.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CONTRACT_VERSION = "neuronlab.run.v1"
KNOWN_RESULT_NAMES = {
    "results.json", "probe-results.json", "experiment.json",
    "report.json", "steering_report.json", "flight_school.json", "workflow.json",
}


def _manifest_status(manifest: Any) -> tuple[str, list[str]]:
    warnings: list[str] = []
    if not isinstance(manifest, dict):
        return "legacy", ["Missing manifest; provenance is incomplete."]
    version = manifest.get("schema_version")
    run_id = manifest.get("run_id")
    if version != CONTRACT_VERSION:
        warnings.append(f"Unsupported or missing schema_version: {version!r}")
    if not run_id:
        warnings.append("Manifest has no run_id.")
    return ("valid" if version == CONTRACT_VERSION and run_id else "legacy"), warnings


def inspect_artifact(path: str | Path) -> dict[str, Any]:
    """Load and normalize one result JSON file; never modifies source artifacts."""
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Artifact root must be a JSON object.")
    is_workflow = source.name == "workflow.json"
    manifest = payload.get("manifest")
    if is_workflow:
        warnings = []
        workflow_version = payload.get("schema_version")
        workflow_id = payload.get("workflow_id")
        status = "valid" if workflow_version == "neuronlab.workflow.v1" and workflow_id else "legacy"
        if status == "legacy":
            warnings.append("Missing or unsupported unified workflow schema/ID.")
        manifest = {
            "project": "neuronlab-unified-workflow",
            "model_id": payload.get("model_id"),
            "run_id": workflow_id,
            "created_at_utc": payload.get("created_at_utc"),
            "schema_version": workflow_version,
        }
    else:
        status, warnings = _manifest_status(manifest)
        if not isinstance(manifest, dict):
            manifest = {}
    rows = payload.get("cases")
    if not isinstance(rows, list):
        rows = payload.get("results")
    if not isinstance(rows, list):
        rows = payload.get("layers")
    if not isinstance(rows, list):
        rows = payload.get("interventions")
    if not isinstance(rows, list):
        rows = payload.get("stages")
    if not isinstance(rows, list):
        rows = [{"id": "baseline", "response": payload["baseline"]}] if isinstance(payload.get("baseline"), str) else []
    return {
        "path": str(source),
        "artifact_name": source.name,
        "project": manifest.get("project") or payload.get("project") or "unknown",
        "model_id": payload.get("model_id") or payload.get("model") or manifest.get("model_id") or "unknown",
        "run_id": manifest.get("run_id"),
        "created_at_utc": manifest.get("created_at_utc"),
        "schema_version": manifest.get("schema_version"),
        "manifest_status": status,
        "warnings": warnings,
        "row_count": len(rows),
        "payload": payload,
    }


def extract_numeric_metrics(payload: dict[str, Any], *, max_metrics: int = 250) -> dict[str, int | float]:
    """Extract comparable numeric scalars without assuming tool-specific result schemas.

    Manifest/provenance values are deliberately excluded. Nested dictionary paths
    become dotted keys; list rows are not expanded because row schemas differ by tool.
    Booleans are excluded because their meaning is not consistently numeric.
    """
    metrics: dict[str, int | float] = {}

    def visit(value: Any, prefix: str = "") -> None:
        if len(metrics) >= max_metrics or not isinstance(value, dict):
            return
        for key, child in value.items():
            if key == "manifest":
                continue
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(child, dict):
                visit(child, path)
            elif isinstance(child, (int, float)) and not isinstance(child, bool):
                metrics[path] = child
            if len(metrics) >= max_metrics:
                return

    visit(payload)
    return metrics


def scan_artifacts(roots: list[str | Path]) -> dict[str, Any]:
    """Scan known result filenames under supplied roots and report duplicates/errors."""
    found: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    seen_paths: set[Path] = set()
    for root_value in roots:
        root = Path(root_value).expanduser()
        if root.is_file():
            candidates = [root] if root.name in KNOWN_RESULT_NAMES else []
        elif root.is_dir():
            candidates = sorted(p for p in root.rglob("*.json") if p.name in KNOWN_RESULT_NAMES)
        else:
            errors.append({"path": str(root), "error": "Path does not exist."})
            continue
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            try:
                found.append(inspect_artifact(resolved))
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                errors.append({"path": str(resolved), "error": str(exc)})
    run_ids: dict[str, list[str]] = {}
    for item in found:
        if item["run_id"]:
            run_ids.setdefault(item["run_id"], []).append(item["path"])
    duplicates = [{"run_id": rid, "paths": paths} for rid, paths in sorted(run_ids.items()) if len(paths) > 1]
    found.sort(key=lambda item: (item["created_at_utc"] or "", item["project"], item["path"]), reverse=True)
    return {
        "contract_version": CONTRACT_VERSION,
        "artifacts": found,
        "errors": errors,
        "duplicate_run_ids": duplicates,
        "summary": {
            "artifacts": len(found),
            "valid_manifests": sum(x["manifest_status"] == "valid" for x in found),
            "legacy_manifests": sum(x["manifest_status"] == "legacy" for x in found),
            "read_errors": len(errors),
            "duplicate_run_ids": len(duplicates),
        },
    }
