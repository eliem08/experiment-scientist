# NeuronLab Unified Workflow Contract

**Schema:** `neuronlab.workflow.v1`  
**JSON Schema:** [`schemas/neuronlab.workflow.v1.schema.json`](schemas/neuronlab.workflow.v1.schema.json)\n\nThe report also records a `reproducibility` block with the Python/runtime platform, SHA-256 and size of the exact case suite, and SHA-256 fingerprints of each evaluator script. These fingerprints identify inputs and code but do not guarantee bit-for-bit deterministic model outputs.

A workflow is one invocation of `unified_evaluation.py`. It is separate from each tool's `neuronlab.run.v1` manifest: the workflow groups multiple runs, while each child artifact keeps its own unique `run_id`.

## Required top-level fields

| Field | Meaning |
|---|---|
| `schema_version` | Fixed workflow schema identifier |
| `workflow_id` | UUID shared by the child manifests |
| `created_at_utc`, `completed_at_utc` | Workflow start and finish timestamps |
| `model_id`, `device`, `config` | Requested common evaluation settings |
| `status` | `completed` when every stage produced a readable artifact; otherwise `failed` |
| `quality_gate` | `passed` plus a list of blocking stage or regression-gate names |
| `stages` | Ordered stage execution records |
| `artifacts` | Paths to result files that exist |
| `report_path` | Path to the saved workflow JSON report |
| `html_report_path` | Path to the self-contained human-readable HTML report |

## Stage contract

Each stage records its name, status, command, expected artifact path, log path, elapsed seconds and exit code when available. A successful process without its expected JSON artifact is a failed stage. Timeouts and process-launch errors are recorded, not silently swallowed. Remaining stages still run after an individual stage fails so a partial report can help diagnose the workflow.

The three built-in stages are:

- **Behavior Bench:** configured regression checks, forbidden hits, expected-check pass rate, and its configured quality gate.
- **Safety Probe Lab:** probe count and marker-based heuristic signals. These signals are triage indicators, not validated safety metrics.
- **Experiment Scientist:** paired-prompt count and descriptive treatment-minus-baseline word delta. Length change is not answer quality.

## Gate semantics

The unified gate fails if any stage fails or if Behavior Bench reports its configured quality gate as failed. A workflow may have `status=completed` and `quality_gate.passed=false`: that means all stages ran, but the results did not pass the gate. The CLI exits with code 0 only when the unified gate passes.

## Provenance

The orchestrator sets `NEURONLAB_WORKFLOW_ID` and `NEURONLAB_WORKFLOW_SCHEMA` in each child process. It also checks that every result's workflow ID and model ID match the current invocation before counting the artifact as successful. The three built-in tools copy these into their manifests while retaining unique run IDs. Standalone invocations leave workflow fields empty.

## Baseline-versus-candidate comparison

Use `compare_workflows.py` to compare two saved workflow reports. It writes a `neuronlab.workflow-comparison.v1` JSON report, described by [`schemas/neuronlab.workflow-comparison.v1.schema.json`](schemas/neuronlab.workflow-comparison.v1.schema.json). The default gate blocks a drop in Behavior Bench expected pass rate greater than 0.05 or an increase in forbidden-hit cases. Thresholds are explicit CLI options. The comparison warns when configs, model IDs, or case-suite fingerprints differ. Safety heuristic counts and response-length deltas are diagnostics only and do not affect the gate.

## Data and security

The workflow is local-first. It writes full model responses into the tools' normal artifacts and local logs; treat the output directory as potentially sensitive. It does not upload results, call hosted APIs, or claim that a successful run certifies model correctness or safety.
