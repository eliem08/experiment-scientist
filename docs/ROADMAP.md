# Research platform roadmap

This roadmap is prioritized by scientific value and risk reduction, not visual novelty.

## Phase A — Stabilize the foundation

- [x] Document the current architecture and known scientific limitations.
- [x] Add a provider-neutral evaluator contract and auditable deterministic evaluators.
- [x] Add independent-unit paired analysis utilities and multiple-comparison correction.
- [ ] Reconcile open revision-pinning/resume work with the current default branch and validate CI on the merged result.
- [ ] Add schema validation for the existing matrix result format and define an explicit compatibility policy.
- [ ] Establish a documented test baseline and run full CI on the integrated branch.

## Phase B — Research core

- [ ] Promote the experiment spec to a complete versioned protocol: datasets, conditions, evaluator versions, analysis plan, exclusions, budgets, and failure policy.
- [ ] Add registration records that distinguish prospective registration from retrospective hashing.
- [ ] Integrate the standalone statistical analysis layer into matrix summaries without changing existing outputs unexpectedly.
- [ ] Add evaluator registry/discovery and schema-validated result persistence.
- [ ] Add calibrated LLM-as-a-judge workflows, human annotations, agreement analysis, and bias checks.
- [ ] Add synthetic fixtures large enough to test strata, missingness, and multiple outcomes.

## Phase C — Reproducible workflow and inspection

- [ ] Verify model/tokenizer revision pinning, environment metadata, and pair-boundary resume.
- [ ] Add atomic writes and stronger crash/corruption tests.
- [ ] Add provenance-aware matrix-to-matrix comparisons with explicit incompatibility and inconclusive states.
- [ ] Extend the Workbench with per-case inspection, aligned outcome plots, uncertainty intervals, and evaluator disagreement.
- [ ] Add exportable analysis figures and machine-readable analysis records.

## Phase D — Reliability and adoption

- [ ] Add dependency/security review, CLI contract tests, and compatibility fixtures for legacy artifacts.
- [ ] Add budget preflight and resource-aware limits.
- [ ] Validate CPU and optional small-model integration paths in CI without requiring large downloads in unit tests.
- [ ] Document clean-checkout reproduction, dataset governance, privacy, retention, and evaluator limitations.
- [ ] Publish a reproducible example study and clearly label synthetic results.

## Acceptance principles

- Never turn missing or failed scores into zero without an explicit, justified rule.
- Never call a marker heuristic a validated safety metric.
- Never treat repeated generations of one case as independent task examples.
- Never compare metrics without checking their definitions and compatible data.
- Never claim that a run is reproducible, a model is safe, or a result is publishable without the evidence needed for that claim.
