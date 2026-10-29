"""Shared reproducibility metadata for a research run."""
from __future__ import annotations
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import hashlib
import os
import platform
import sys
import uuid

def _package_version(name: str):
    try:
        return version(name)
    except PackageNotFoundError:
        return None

def build_manifest(project: str, *, model_id=None, device=None, seed=None, config=None, inputs=None) -> dict:
    """Return a JSON-serializable manifest without requiring optional ML packages."""
    input_hashes = {}
    for name, value in (inputs or {}).items():
        raw = value if isinstance(value, bytes) else str(value).encode("utf-8")
        input_hashes[name] = hashlib.sha256(raw).hexdigest()
    return {
        "schema_version": "neuronlab.run.v1",
        "run_id": str(uuid.uuid4()),
        "project": project,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": os.environ.get("GITHUB_SHA"),
        "model_id": model_id,
        "device": device,
        "seed": seed,
        "package_versions": {
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "numpy": _package_version("numpy"),
            "streamlit": _package_version("streamlit"),
        },
        "config": config or {},
        "input_sha256": input_hashes,
    }
