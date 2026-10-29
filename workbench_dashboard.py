"""NeuronLab Workbench MVP: browse, filter, compare, and export local run artifacts.

Run with: streamlit run workbench_dashboard.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from artifact_registry import extract_numeric_metrics, scan_artifacts
from workbench_analytics import compare_metric_maps, summarize_artifact

st.set_page_config(page_title="NeuronLab Workbench", layout="wide")
st.title("NeuronLab Workbench")
st.caption("Local-first run history and cross-run comparison for all six NeuronLab tools.")
st.info("Artifacts are read-only. This dashboard does not upload results or execute models.")

default_root = str(Path("outputs").resolve())
roots_text = st.text_area(
    "Artifact folders or result JSON paths (one per line)",
    value=default_root,
    help="Point this at a shared folder containing outputs from any NeuronLab tool. You can provide multiple folders or exact supported result JSON files.",
)
roots = [line.strip() for line in roots_text.splitlines() if line.strip()]
if st.button("Scan run artifacts", type="primary"):
    st.session_state["neuronlab_scan"] = scan_artifacts(roots)
result = st.session_state.get("neuronlab_scan")
if result is None:
    st.write("Enter one or more local output folders, then scan to build a run history.")
    st.stop()

summary = result["summary"]
cols = st.columns(5)
for col, label, key in zip(
    cols,
    ["Artifacts", "Valid manifests", "Legacy manifests", "Read errors", "Duplicate run IDs"],
    ["artifacts", "valid_manifests", "legacy_manifests", "read_errors", "duplicate_run_ids"],
):
    col.metric(label, summary[key])

if result["errors"]:
    with st.expander("Read errors", expanded=True):
        st.dataframe(pd.DataFrame(result["errors"]), use_container_width=True)
if result["duplicate_run_ids"]:
    with st.expander("Duplicate run IDs", expanded=True):
        st.dataframe(pd.DataFrame(result["duplicate_run_ids"]), use_container_width=True)

artifacts = result["artifacts"]
if not artifacts:
    st.warning("No supported result artifacts found. Supported files include results.json, probe-results.json, experiment.json, report.json, steering_report.json, and flight_school.json.")
    st.stop()

st.subheader("Run history")
filter_cols = st.columns(3)
projects = sorted({a["project"] for a in artifacts})
models = sorted({a["model_id"] for a in artifacts})
with filter_cols[0]:
    project_filter = st.selectbox("Project", ["All projects", *projects])
with filter_cols[1]:
    model_filter = st.selectbox("Model", ["All models", *models])
with filter_cols[2]:
    manifest_filter = st.selectbox("Manifest status", ["All statuses", "valid", "legacy"])

visible = [
    a for a in artifacts
    if (project_filter == "All projects" or a["project"] == project_filter)
    and (model_filter == "All models" or a["model_id"] == model_filter)
    and (manifest_filter == "All statuses" or a["manifest_status"] == manifest_filter)
]
table_rows = [{
    "project": a["project"],
    "model_id": a["model_id"],
    "created_at_utc": a["created_at_utc"] or "unknown",
    "run_id": a["run_id"] or "missing",
    "manifest": a["manifest_status"],
    "rows": a["row_count"],
    "file": a["artifact_name"],
    "path": a["path"],
} for a in visible]
st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
if not visible:
    st.warning("No artifacts match the selected filters.")
    st.stop()

history_csv = pd.DataFrame(table_rows).to_csv(index=False).encode("utf-8")
st.download_button("Export filtered run history (CSV)", history_csv, "neuronlab-run-history.csv", "text/csv")

st.subheader("Inspect an artifact")
labels = [f'{a["project"]} · {a["model_id"]} · {a["artifact_name"]} · {a["path"]}' for a in visible]
selected = st.selectbox("Artifact details", labels)
artifact = visible[labels.index(selected)]
st.json({key: artifact[key] for key in ("project", "model_id", "run_id", "created_at_utc", "schema_version", "manifest_status", "warnings")})
if artifact["project"] == "neuronlab-unified-workflow":
    workflow_payload = artifact["payload"]
    gate = workflow_payload.get("quality_gate", {})
    st.subheader("Unified workflow stages")
    st.metric("Unified quality gate", "PASS" if gate.get("passed") is True else "FAIL")
    stage_rows = []
    for stage in workflow_payload.get("stages", []):
        stage_rows.append({
            "stage": stage.get("name"),
            "status": stage.get("status"),
            "exit_code": stage.get("exit_code"),
            "duration_seconds": stage.get("duration_seconds"),
            "artifact": stage.get("artifact"),
            "error": stage.get("error", ""),
        })
    if stage_rows:
        st.dataframe(pd.DataFrame(stage_rows), use_container_width=True, hide_index=True)
    if gate.get("blockers"):
        st.warning("Quality-gate blockers: " + ", ".join(str(item) for item in gate["blockers"]))
with st.expander("Raw JSON artifact"):
    st.code(json.dumps(artifact["payload"], indent=2, ensure_ascii=False), language="json")

st.subheader("Project-aware scorecards")
st.caption("Metrics are interpreted only where the source artifact has a known structure. Heuristic safety signals and substring checks are not certifications.")
scorecard_rows = []
for item in visible:
    for metric in summarize_artifact(item):
        scorecard_rows.append({
            "project": item["project"],
            "model_id": item["model_id"],
            "run_id": item["run_id"] or item["artifact_name"],
            "metric": metric["metric"],
            "value": metric["value"],
            "interpretation": metric["note"],
        })
if scorecard_rows:
    st.dataframe(pd.DataFrame(scorecard_rows), use_container_width=True, hide_index=True)
else:
    st.info("No project-specific scorecard fields found in the selected artifacts.")

st.subheader("Cross-run comparison")
st.caption("Choose two or more artifacts. Only numeric scalar fields with matching dotted paths are compared; row-level data is not coerced into a shared schema.")
compare_labels = st.multiselect(
    "Runs to compare",
    options=labels,
    default=labels[: min(2, len(labels))],
)
selected_runs = [visible[labels.index(label)] for label in compare_labels]
if len(selected_runs) < 2:
    st.info("Select at least two runs to build a comparison.")
else:
    metric_maps = [extract_numeric_metrics(a["payload"]) for a in selected_runs]
    common_metrics = sorted(set.intersection(*(set(m) for m in metric_maps))) if metric_maps else []
    run_names = [
        f'{a["project"]} | {a["model_id"]} | {a["run_id"] or a["artifact_name"]}'
        for a in selected_runs
    ]
    metadata = pd.DataFrame([
        {
            "run": run_names[i],
            "created_at_utc": a["created_at_utc"] or "unknown",
            "manifest_status": a["manifest_status"],
            "numeric_metrics_found": len(metric_maps[i]),
        }
        for i, a in enumerate(selected_runs)
    ])
    st.dataframe(metadata, use_container_width=True, hide_index=True)
    if common_metrics:
        comparison = pd.DataFrame(compare_metric_maps(metric_maps, run_names))
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        st.download_button(
            "Export metric comparison (CSV)",
            comparison.to_csv(index=False).encode("utf-8"),
            "neuronlab-metric-comparison.csv",
            "text/csv",
        )
    else:
        st.warning("These artifacts have no matching numeric scalar fields outside the manifest. Inspect the raw JSON or compare runs from the same tool.")
