# Six-Tool Orchestration: Implementation Contract

**Status:** design contract for the next implementation slice  
**Target:** extend the existing unified evaluation workflow without weakening its current provenance, failure-reporting, or quality-gate behavior.

## Current verified baseline

The current `unified_evaluation.py` workflow executes three local Transformers evaluators:

1. Behavior Bench
2. Safety Probe Lab
3. Experiment Scientist

The Workbench artifact registry can inspect artifacts from all six NeuronLab tools, but artifact discovery is not the same as executing all six tools. The existing workflow contract must remain backward-compatible while execution coverage expands.

## Goals

- Run each configured tool through an explicit adapter, not shell-string guessing.
- Keep each tool's native artifact intact and record its path in the parent workflow.
- Add a shared workflow ID and schema version to child provenance where the tool supports it; otherwise record the limitation explicitly in the stage record.
- Continue later stages after a stage failure, preserving partial results and logs.
- Distinguish execution status from metric quality and from tool availability.
- Keep API-backed stages opt-in. Never require secrets for the local-only default, and never print credentials into reports or logs.
- Never compare incomparable metrics or present heuristic indicators as validated safety/correctness scores.

## Stage record contract

Every configured stage should have these fields, where available:

- `name`, `kind` (`local_command` or `api`), `status`
- sanitized argument list or a redacted API operation description
- `started_at_utc`, `completed_at_utc`, `elapsed_seconds`
- `exit_code`, `error`, `log_path`
- `expected_artifacts`, `artifacts`
- `workflow_id_propagated` (boolean) and `provenance_warnings`
- `metrics` with source/meaning metadata, never an unlabelled mixed metric

Use a finite status vocabulary: `pending`, `running`, `completed`, `failed`, `timed_out`, `skipped`, `unavailable`. A missing optional integration should be `unavailable` or `skipped`, not silently reported as completed.

## Implementation sequence

### Phase 1 — adapter registry and preflight

- Define a declarative stage registry with required paths, commands, environment requirements, expected artifacts, timeout, and whether the stage is enabled by default.
- Validate all configured local paths and executable entry points before expensive model downloads.
- Support an explicit selected-stage list so a developer can smoke-test one adapter without pretending the full suite ran.
- Keep the existing three-stage invocation and report shape compatible.

**Acceptance:** unit tests cover valid configuration, missing paths, invalid stage names, duplicate stage names, and API stages disabled by default.

### Phase 2 — remaining local tools

- Identify the exact remaining tool repositories and their documented invocation/artifact contracts before implementing adapters.
- Add one adapter per tool; do not assume every tool is a Transformers evaluator.
- Preserve each tool's native artifact and add a small normalized stage summary only.
- Add synthetic-fixture tests for success, non-zero exit, missing artifact, malformed JSON, timeout, and workflow-ID mismatch.

**Acceptance:** a six-stage run lists all six configured tools. A tool that is not installed is explicitly unavailable and causes the configured gate to fail unless the user explicitly selected a partial run.

### Phase 3 — API-backed integration

- Add API-backed adapters only after the provider, request semantics, data handling, rate limits, and output schema are specified.
- Require explicit opt-in and environment-based credentials.
- Redact authorization headers, tokens, and sensitive request fields from logs and reports.
- Store request/response payloads only according to an explicit retention setting; never upload local artifacts by default.

**Acceptance:** tests verify missing credentials, provider errors, rate limits, redaction, and opt-in behavior without making live network calls.

### Phase 4 — cross-tool report and quality gate

- Keep per-tool metrics separate by default.
- Permit cross-tool comparisons only when a documented mapping establishes compatible definitions, units, datasets, and directionality.
- Report stage execution, artifact validity, provenance completeness, and quality-gate outcome as separate dimensions.
- Keep HTML output self-contained and ensure paths are clearly marked as local artifacts.

**Acceptance:** regression tests prove that a completed run may still fail the quality gate, failed stages remain visible, and diagnostic heuristics do not silently become certification claims.

## Non-goals

- No forced common schema for full model responses.
- No automatic uploading of artifacts.
- No claim that stage completion establishes model safety, correctness, or scientific validity.
- No hard-coded secrets or automatic API execution.
- No speculative adapters for unidentified repositories.

## Immediate next action

Inspect the remaining three repositories and record each one's real command, dependency set, output artifact, and provenance support. Then implement and test adapters individually before enabling a six-stage default run.
