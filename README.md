# Experiment Scientist

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eliem08/experiment-scientist/blob/main/colab.ipynb)

A paired-prompt experiment runner that compares baseline and treatment instructions with fixed greedy decoding.

## Interactive dashboard

    python -m pip install -r requirements.txt
    streamlit run dashboard.py

Launch a paired experiment and compare word counts and individual baseline/treatment responses.

## Run an experiment

    python prompt_experiment.py --model-id distilgpt2 --device auto --seed 17

Writes JSON and CSV with prompts, responses, word counts, hashes, and treatment-minus-baseline word deltas.

The built-in examples are demonstrations, not a pre-registered study. Define hypotheses and outcomes in advance, use enough diverse prompts, track model/tokenizer versions, and test across multiple seeds or sampling settings where relevant.

## License

MIT.

## NeuronLab Workbench (local artifact history)

The integration shell can browse, filter, summarize, compare, and export result artifacts from all six NeuronLab tools using the shared `neuronlab.run.v1` manifest. It is read-only and keeps artifacts on the machine; it does not upload data or run models. Install the lightweight Workbench dependencies separately so you do not need to install PyTorch or Transformers just to inspect existing results.

    python -m pip install -r requirements-workbench.txt
    streamlit run workbench_dashboard.py

Enter one or more local output folders (or exact supported result JSON paths), then scan. The registry reports run IDs, model IDs, timestamps, manifest compatibility, malformed artifacts, and duplicate run IDs. Supported filenames include `results.json`, `probe-results.json`, `experiment.json`, `report.json`, `steering.json`, `steering_report.json`, `flight_school.json`, and `depth_report.json`. Neuron Explorer's precomputed `depth_report.json` is imported as a legacy artifact because its GPU-backed sweep is not a single-run local evaluator; the Workbench does not launch that sweep. Older artifacts remain visible but are marked as legacy if they lack the shared manifest.

For a combined history, place each tool's output directory under a common local folder and scan that folder. No cross-tool result rows are forced into one schema; the registry normalizes run provenance while preserving the raw JSON for inspection.


## Headless Workbench CLI

You can scan artifacts without launching Streamlit, which is useful for automation and CI:

    python workbench_cli.py outputs
    python workbench_cli.py outputs --format csv --output run-history.csv
    python workbench_cli.py outputs --format json --output scan-report.json

The CLI exports normalized run metadata, manifest warnings, read errors, and duplicate run IDs without copying full model responses into the report. It uses the Python standard library only. It exits with status 1 if any artifact could not be read, making it suitable for a basic pipeline quality check.


## Unified evaluation workflow

The unified runner always executes the three core local evaluators (Behavior Bench, Safety Probe Lab, and Experiment Scientist). It can additionally execute Steering Studio and Neural Flight School when their repository paths are supplied. Each tool keeps its native artifact; the runner links the outputs in `workflow.json` and a self-contained `workflow-report.html`. Optional tools are not silently treated as complete when their paths are omitted. The sixth tool, Neuron Explorer, is integrated into artifact history through precomputed `depth_report.json` files. Its GPU-backed sweep is deliberately not launched automatically; use its own documented sweep workflow, then point the Workbench at the resulting directory. A failed stage stays visible, later stages still run, and the quality gate fails.

Clone the repositories as sibling folders next to `experiment-scientist`. On Windows, `./setup_unified_evaluation.ps1` clones the two core sibling repositories if missing, creates `.venv`, and installs the runtime dependencies from `requirements-evaluation.txt`. Use `-SkipInstall` to clone only. On macOS/Linux, clone the repositories manually. Then run from this repository:

    python -m venv .venv
    # Windows PowerShell: .\.venv\Scripts\Activate.ps1
    # macOS/Linux: source .venv/bin/activate
    python -m pip install -r requirements-evaluation.txt
    python unified_evaluation.py --model-id distilgpt2 --device cpu

# Optional research adapters (requires both sibling repositories)
python unified_evaluation.py --model-id gpt2 --device cpu --steering-studio-dir ../steering-studio --neural-flight-school-dir ../neural-flight-school

On Windows PowerShell, use the same command, or run `./run_unified_evaluation.ps1 --model-id distilgpt2 --device cpu`. Steering Studio defaults to transformer layer 6; use `--steering-layer 1` for small GPT-2 variants that have fewer layers.

To customize paths or output:

    python unified_evaluation.py --behavior-bench-dir ../behavior-bench --safety-probe-dir ../safety-probe-lab --experiment-scientist-dir . --model-id distilgpt2 --device auto --output outputs/unified-evaluation

The default case suite is `behavior-bench/examples/cases.jsonl`. Model weights may need to be downloaded on the first run, and the workflow can take a while on CPU. The default stage timeout is two hours; change it with `--timeout-seconds`.

See [WORKFLOW_CONTRACT.md](WORKFLOW_CONTRACT.md) for the machine-readable workflow semantics and gate rules. You can pass `--minimum-failing-severity high` to configure Behavior Bench's regression threshold. The unified quality gate fails if any stage fails or if Behavior Bench's configured regression gate fails. Safety Probe Lab's marker signals are shown as heuristic triage, not as a validated safety score; Experiment Scientist's word deltas are descriptive response-length differences, not measures of answer quality. The workflow does not claim a model is safe or correct simply because its stages completed.

The top-level `workflow.json` is also recognized by the Workbench run-history registry. Point `workbench_dashboard.py` at the workflow output directory to inspect the unified workflow alongside individual run artifacts.


### Compare a candidate run against a baseline

After running the workflow for a baseline model or revision, keep its `workflow.json` and compare a candidate workflow against it:

```powershell
python compare_workflows.py --baseline outputs/baseline/workflow.json --candidate outputs/unified-evaluation/workflow.json --output outputs/workflow-comparison.json
```

The comparison writes both machine-readable JSON and a self-contained HTML report (by default, `workflow-comparison.json` and `workflow-comparison.html`). Use `--html-output path/to/report.html` to choose the HTML destination. The HTML report includes metric deltas, configuration/model mismatch warnings, gate blockers, and interpretation limits. By default, it blocks an expected-pass-rate drop greater than 5 percentage points or any increase in Behavior Bench cases with forbidden-marker hits. Tune thresholds explicitly with `--max-pass-rate-drop 0.02` and `--max-forbidden-hit-increase 1`. It exits non-zero when a configured regression is detected or the candidate workflow's own quality gate failed.

Safety Probe Lab heuristic counts and Experiment Scientist response-length deltas are diagnostics only; they do not affect the regression gate. Keep baseline and candidate case suites and metric definitions comparable. This is an engineering regression signal, not a certification of correctness or safety. Regression comparisons are tested with synthetic fixtures in CI; run actual local model inference before relying on results.


### Real-model smoke run in GitHub Actions

The workflow [`unified-smoke.yml`](.github/workflows/unified-smoke.yml) runs automatically on pull requests and can also be started manually; it runs all five executable stages against an actual Hugging Face model on CPU, validates that every stage completed, and uploads the JSON/HTML report plus logs as a workflow artifact. It checks out Steering Studio and Neural Flight School as well as the three core repositories. To run it manually, open the repository's **Actions** tab, select **unified-evaluation-smoke**, click **Run workflow**, and keep the default `sshleifer/tiny-gpt2` model for the smallest initial download. The regression/quality gate may legitimately fail on a tiny model; the smoke check distinguishes that from an evaluator execution failure. This manual run has to be triggered from GitHub and has not been claimed as completed until its artifact is available.


## Pre-registered repeated-seed experiment matrix

The original \`prompt_experiment.py\` remains a quick, fixed example. For a more rigorous study, define the hypothesis, primary outcome, scoring rules, and paired prompts in a JSON spec **before running inference**. The new matrix runner validates the spec, records its exact SHA-256, evaluates multiple models and seeds, preserves every paired observation, and reports case-level paired bootstrap intervals plus paired effect size.

\`\`\`bash
python experiment_matrix.py \\
  --spec examples/uncertainty-ablation.spec.json \\
  --model-id distilgpt2 \\
  --model-id gpt2 \\
  --seeds 17,29,43,71,101 \\
  --device auto \\
  --output outputs/uncertainty-ablation
\`\`\`

Use \`--sample\` to enable seeded sampling (defaults are greedy decoding). Results are written to \`matrix-results.json\` and \`matrix-observations.csv\`. Supported transparent outcome functions are \`word_count\`, \`character_count\`, \`contains\`, \`regex_match\`, and \`exact_match\`; the spec must declare which metric is primary. See [the experiment matrix guide](docs/EXPERIMENT_MATRIX.md) for the schema, interpretation, and limitations.

**Scientific caution:** these deterministic metrics do not judge semantic correctness or safety. Repeated seeds do not compensate for too few or unrepresentative prompts. Bootstrap intervals are computed over case-level paired deltas after averaging seeds, avoiding the mistake of treating repeated runs of one prompt as independent prompts. The saved spec hash supports auditability, but does not independently prove when the spec was authored.


### Performance measurements and publishable summary files

The matrix runner now records input/output token counts and generation latency for each baseline and treatment arm, plus per-model mean latency and output-token throughput. These are runtime measurements, not a hosted-API price estimate. Reports avoid duplicating pair-level performance statistics just because multiple outcome metrics are declared.

Generate human-readable Markdown and HTML reports from a completed result:

\`\`\`bash
python render_matrix_report.py --input outputs/uncertainty-ablation/matrix-results.json
\`\`\`

By default this writes `matrix-results.md` and `matrix-results.html` beside the input file. Use `--output-md` and `--output-html` to choose destinations. The reports include the hypothesis and spec fingerprint, outcome summaries and intervals, performance comparisons, resolved revisions/runtime metadata when available, and interpretation limits. Raw observations remain in JSON/CSV rather than being copied into the summary report.
