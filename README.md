# Experiment Scientist

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eliem08/experiment-scientist/blob/main/colab.ipynb)

A paired-prompt experiment runner that compares baseline and treatment instructions with fixed greedy decoding.

## Interactive dashboard

    pip install -r requirements.txt
    streamlit run dashboard.py

Launch a paired experiment and compare word counts and individual baseline/treatment responses.

## Run an experiment

    python prompt_experiment.py --model-id distilgpt2 --device auto --seed 17

Writes JSON and CSV with prompts, responses, word counts, hashes, and treatment-minus-baseline word deltas.

The built-in examples are demonstrations, not a pre-registered study. Define hypotheses and outcomes in advance, use enough diverse prompts, track model/tokenizer versions, and test across multiple seeds or sampling settings where relevant.

## License

MIT.

## NeuronLab Workbench (local artifact history)

The first integration shell can browse result artifacts from all six NeuronLab tools using the shared `neuronlab.run.v1` manifest. It is read-only and keeps artifacts on the machine; it does not upload data or run models.

    streamlit run workbench_dashboard.py

Enter one or more local output folders (or exact supported result JSON paths), then scan. The registry reports run IDs, model IDs, timestamps, manifest compatibility, malformed artifacts, and duplicate run IDs. Supported filenames include `results.json`, `probe-results.json`, `experiment.json`, `report.json`, `steering_report.json`, and `flight_school.json`. Older artifacts remain visible but are marked as legacy if they lack the shared manifest.

For a combined history, place each tool's output directory under a common local folder and scan that folder. No cross-tool result rows are forced into one schema; the registry normalizes run provenance while preserving the raw JSON for inspection.


## Headless Workbench CLI

You can scan artifacts without launching Streamlit, which is useful for automation and CI:

    python workbench_cli.py outputs
    python workbench_cli.py outputs --format csv --output run-history.csv
    python workbench_cli.py outputs --format json --output scan-report.json

The CLI exports normalized run metadata, manifest warnings, read errors, and duplicate run IDs without copying full model responses into the report. It exits with status 1 if any artifact could not be read, making it suitable for a basic pipeline quality check.
