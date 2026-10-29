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

Enter one or more local output folders (or exact supported result JSON paths), then scan. The registry reports run IDs, model IDs, timestamps, manifest compatibility, malformed artifacts, and duplicate run IDs. Supported filenames include `results.json`, `probe-results.json`, `experiment.json`, `report.json`, `steering_report.json`, and `flight_school.json`. Older artifacts remain visible but are marked as legacy if they lack the shared manifest.

For a combined history, place each tool's output directory under a common local folder and scan that folder. No cross-tool result rows are forced into one schema; the registry normalizes run provenance while preserving the raw JSON for inspection.


## Headless Workbench CLI

You can scan artifacts without launching Streamlit, which is useful for automation and CI:

    python workbench_cli.py outputs
    python workbench_cli.py outputs --format csv --output run-history.csv
    python workbench_cli.py outputs --format json --output scan-report.json

The CLI exports normalized run metadata, manifest warnings, read errors, and duplicate run IDs without copying full model responses into the report. It uses the Python standard library only. It exits with status 1 if any artifact could not be read, making it suitable for a basic pipeline quality check.


## Unified evaluation workflow

The unified runner orchestrates the local model evaluators in Behavior Bench, Safety Probe Lab, and Experiment Scientist, using the same model ID, device, and generation-token limit. Each tool still writes its own native artifact and log; the runner adds a shared workflow ID to each provenance manifest and creates a top-level `workflow.json` with stage statuses, artifact paths, key metrics, and a consolidated quality gate. A failed stage does not silently disappear: later stages still run, and the workflow exits non-zero.

Clone the three repositories as sibling folders next to `experiment-scientist`. Create one shared Python virtual environment and install the common requirements once (the three `requirements.txt` files currently match), then run from this repository:

    python -m venv .venv
    # Windows PowerShell: .\.venv\Scripts\Activate.ps1
    # macOS/Linux: source .venv/bin/activate
    python -m pip install -r requirements.txt
    python unified_evaluation.py --model-id distilgpt2 --device cpu

To customize paths or output:

    python unified_evaluation.py --behavior-bench-dir ../behavior-bench --safety-probe-dir ../safety-probe-lab --experiment-scientist-dir . --model-id distilgpt2 --device auto --output outputs/unified-evaluation

The default case suite is `behavior-bench/examples/cases.jsonl`. Model weights may need to be downloaded on the first run, and the workflow can take a while on CPU. The default stage timeout is two hours; change it with `--timeout-seconds`.

See [WORKFLOW_CONTRACT.md](WORKFLOW_CONTRACT.md) for the machine-readable workflow semantics and gate rules. You can pass `--minimum-failing-severity high` to configure Behavior Bench's regression threshold. The unified quality gate fails if any stage fails or if Behavior Bench's configured regression gate fails. Safety Probe Lab's marker signals are shown as heuristic triage, not as a validated safety score; Experiment Scientist's word deltas are descriptive response-length differences, not measures of answer quality. The workflow does not claim a model is safe or correct simply because its stages completed.

The top-level `workflow.json` is also recognized by the Workbench run-history registry. Point `workbench_dashboard.py` at the workflow output directory to inspect the unified workflow alongside individual run artifacts.
