# Architecture assessment

**Status:** incremental assessment of the repository; not a claim that every workflow or hardware path has been independently executed.

## Current shape

Experiment Scientist currently has several related but distinct workflows:

- **Quick experiment:** `prompt_experiment.py` for a small paired prompt comparison.
- **Matrix experiment:** `experiment_matrix.py` validates a JSON spec, executes model × seed × case comparisons, and writes JSON/CSV observations and summaries.
- **Reporting:** `render_matrix_report.py` renders a matrix result as Markdown and HTML.
- **Multi-tool workflow:** `unified_evaluation.py` launches configured evaluator repositories as subprocesses and writes a parent workflow artifact. Native tool outputs remain separate.
- **Regression comparison:** `compare_workflows.py` compares selected workflow-level metrics and applies explicitly configured Behavior Bench gates.
- **Artifact history:** `artifact_registry.py`, `workbench_analytics.py`, `workbench_cli.py`, and `workbench_dashboard.py` discover, summarize, and compare local artifacts.
- **External integrations:** Behavior Bench and Safety Probe Lab are core workflow stages; Steering Studio and Neural Flight School are optional runnable stages; Neuron Explorer is imported from a precomputed artifact rather than launched automatically.

## Strengths

- Local-first artifact handling and explicit interpretation caveats.
- Paired observations, repeated seeds, declared primary metrics, and spec fingerprints.
- Runtime/performance reporting and HTML escaping in the matrix report.
- Clear distinction between successful integration execution and a behavioral quality-gate pass.
- A headless artifact scanner that does not need the Streamlit UI to inspect stored runs.

## Main gaps and risks

1. The built-in matrix outcomes are deterministic string/rule functions; they do not establish semantic correctness or safety.
2. The example matrix spec has only two cases and is a tutorial fixture, not a benchmark suitable for publication.
3. Statistical logic is currently coupled to the matrix runner. A dedicated module improves reuse and independent testing, but its output still needs to be declared in an experiment analysis plan.
4. The system does not yet expose a uniform, versioned evaluator plugin contract for rule-based, task-specific, judge-based, and human annotations.
5. Artifact normalization and workflow-level metric comparison must avoid implying that similarly named scalar fields from different tools have the same meaning.
6. Model execution is resource-intensive and depends on model downloads, runtime libraries, device behavior, and external tool repositories. Unit-test success alone cannot validate those integrations.
7. A spec hash supports content integrity but does not independently prove that a hypothesis was registered before exploratory analysis.
8. Reports and checkpoints may contain prompts and generated text. Treat them as potentially sensitive local research data.

## Target boundaries

- **Experiment specification:** declared conditions, dataset identity, metrics, and analysis plan.
- **Execution:** model loading, generation, checkpointing, runtime telemetry, and observation collection.
- **Evaluation:** versioned evaluator interface and evaluator-specific outputs.
- **Analysis:** paired-unit aggregation, uncertainty, effect sizes, and multiplicity adjustments.
- **Artifacts:** versioned records and provenance; preserve native tool schemas.
- **Presentation:** Workbench, CLI, and report rendering; never infer scientific validity from UI status.

The new `evaluation_core.py` and `statistical_analysis.py` modules introduce the first reusable boundaries for evaluation and analysis. They are additive; they do not silently change the existing matrix artifact schema or replace legacy metrics.
