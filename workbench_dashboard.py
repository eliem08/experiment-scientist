"""NeuronLab Workbench MVP: browse run artifacts from all standalone tools.

Run with: streamlit run workbench_dashboard.py
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from artifact_registry import scan_artifacts

st.set_page_config(page_title="NeuronLab Workbench", layout="wide")
st.title("NeuronLab Workbench")
st.caption("Local-first run history for all six NeuronLab tools.")
st.info("Artifacts are read-only. This dashboard does not upload results or execute models.")
default_root = str(Path("outputs").resolve())
roots_text = st.text_area("Artifact folders or result JSON paths (one per line)", value=default_root,
    help="Point this at a shared folder containing outputs from any NeuronLab tool. You can provide multiple folders or exact supported result JSON files.")
roots = [line.strip() for line in roots_text.splitlines() if line.strip()]
if st.button("Scan run artifacts", type="primary"):
    st.session_state["neuronlab_scan"] = scan_artifacts(roots)
result = st.session_state.get("neuronlab_scan")
if result is None:
    st.write("Enter one or more local output folders, then scan to build a run history.")
    st.stop()
summary = result["summary"]
cols = st.columns(5)
for col, label, key in zip(cols, ["Artifacts", "Valid manifests", "Legacy manifests", "Read errors", "Duplicate run IDs"],
                           ["artifacts", "valid_manifests", "legacy_manifests", "read_errors", "duplicate_run_ids"]):
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
table_rows = [{"project": a["project"], "model_id": a["model_id"],
    "created_at_utc": a["created_at_utc"] or "unknown", "run_id": a["run_id"] or "missing",
    "manifest": a["manifest_status"], "rows": a["row_count"], "file": a["artifact_name"], "path": a["path"]}
    for a in artifacts]
st.subheader("Run history")
st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
labels = [f'{a["project"]} · {a["model_id"]} · {a["artifact_name"]} · {a["path"]}' for a in artifacts]
selected = st.selectbox("Inspect artifact", labels)
artifact = artifacts[labels.index(selected)]
st.subheader("Artifact details")
st.json({key: artifact[key] for key in ("project", "model_id", "run_id", "created_at_utc", "schema_version", "manifest_status", "warnings")})
with st.expander("Raw JSON artifact"):
    st.code(json.dumps(artifact["payload"], indent=2, ensure_ascii=False), language="json")
