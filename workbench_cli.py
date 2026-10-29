"""Headless NeuronLab Workbench scanner for scripts and CI.

Examples:
    python workbench_cli.py outputs
    python workbench_cli.py outputs --format csv --output run-history.csv
    python workbench_cli.py outputs --format json --output scan.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence

from artifact_registry import scan_artifacts

SUMMARY_FIELDS = (
    "project", "model_id", "run_id", "workflow_id", "created_at_utc", "schema_version",
    "manifest_status", "row_count", "artifact_name", "path", "workflow_status", "quality_gate_passed",
)


def build_report(roots: Sequence[str]) -> dict:
    """Return a portable report without embedding raw experiment payloads."""
    result = scan_artifacts(list(roots))
    artifacts = []
    for item in result["artifacts"]:
        payload = item.get("payload", {})
        gate = payload.get("quality_gate") if isinstance(payload, dict) else None
        artifacts.append({key: item.get(key) for key in SUMMARY_FIELDS} | {
            "workflow_status": payload.get("status") if payload.get("schema_version") == "neuronlab.workflow.v1" else None,
            "quality_gate_passed": gate.get("passed") if isinstance(gate, dict) else None,
            "warnings": item["warnings"],
        })
    return {
        "contract_version": result["contract_version"],
        "summary": result["summary"],
        "artifacts": artifacts,
        "errors": result["errors"],
        "duplicate_run_ids": result["duplicate_run_ids"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan local NeuronLab result artifacts.")
    parser.add_argument("roots", nargs="+", help="Directories or supported result JSON files to scan.")
    parser.add_argument("--format", choices=("json", "csv"), default="json", help="Report format.")
    parser.add_argument("--output", help="Optional output file; stdout is used when omitted.")
    args = parser.parse_args(argv)

    report = build_report(args.roots)
    if args.format == "json":
        rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    else:
        import io
        buffer = io.StringIO(newline="")
        fields = list(SUMMARY_FIELDS) + ["warnings"]
        writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for artifact in report["artifacts"]:
            row = dict(artifact)
            row["warnings"] = "; ".join(row.get("warnings") or [])
            writer.writerow(row)
        rendered = buffer.getvalue()

    if args.output:
        Path(args.output).expanduser().write_text(rendered, encoding="utf-8", newline="")
    else:
        sys.stdout.write(rendered)
    return 1 if report["summary"]["read_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
