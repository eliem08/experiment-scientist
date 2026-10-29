# NeuronLab Research Ecosystem

These repositories are separate, composable tools intended to evolve into one local-first AI research workbench.

## The six modules

- [Model MRI](https://github.com/eliem08/model-mri) — inspect hidden-state activation statistics.
- [Behavior Bench](https://github.com/eliem08/behavior-bench) — run prompt regression suites.
- [Safety Probe Lab](https://github.com/eliem08/safety-probe-lab) — probe uncertainty, evidence, privacy, and instruction conflicts.
- [Experiment Scientist](https://github.com/eliem08/experiment-scientist) — compare baseline and treatment conditions.
- [Steering Studio](https://github.com/eliem08/steering-studio) — test controlled activation interventions.
- [Neural Flight School](https://github.com/eliem08/neural-flight-school) — teach transformer internals through interactive visualizations.

## Shared run contract

Every module's JSON artifact includes a `manifest` using schema version `neuronlab.run.v1`. The shared manifest fields are:
- `schema_version`: contract version (`neuronlab.run.v1`).
- `run_id`: unique UUID for this execution.
- `project`, `created_at_utc`, `git_commit`, `python_version`, and `platform`.
- `model_id`, `device`, and `seed` when applicable.
- `package_versions`, `config`, and `input_sha256` for reproducibility.

Each module also saves raw inputs and model outputs where appropriate, JSON/CSV result artifacts, and a warning that distinguishes exploratory diagnostics from validated evaluation. Module-specific result rows may differ, but consumers can reliably identify the run and read provenance from the shared manifest. Contract changes should increment the schema version rather than silently changing the meaning of existing fields.

## Final integrated product: NeuronLab Workbench

A local-first desktop/web app that provides a single model registry, one project/session format, shared experiment history, artifact browser, comparison dashboards, and reproducible report export. The six modules remain usable independently and can also be invoked as jobs by the integrated shell.

### Architecture target

- Frontend: Streamlit first for the research MVP; later a polished web UI if demand justifies it.
- Execution: Python workers with CPU/GPU auto-detection, optional job queue for longer runs.
- Storage: local project folders containing manifests, datasets, logs, and results; no upload by default.
- Model access: Hugging Face local/open-weight models first; API adapters later, with secrets supplied only through environment variables.
- Reproducibility: immutable run IDs, input hashes, model revision, generation config, package versions, and exportable reports.
- Quality gates: unit tests, CI, fixture-based tests, and human review for safety/semantic claims.

## Product principle

Make every result inspectable and reproducible. Do not market keyword heuristics as safety certification, random-weight attention as learned behavior, or one prompt pair as causal evidence.
