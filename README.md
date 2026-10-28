# Experiment Scientist

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eliem08/experiment-scientist/blob/main/colab.ipynb)

A paired-prompt experiment runner that compares baseline and treatment instructions with fixed greedy decoding.

## Run an experiment

    pip install -r requirements.txt
    python prompt_experiment.py --model-id distilgpt2

Writes outputs/experiment/experiment.json and .csv, with both prompts and responses, word counts, hashes, and treatment-minus-baseline word deltas.

    python prompt_experiment.py --model-id distilgpt2 --device cuda --seed 17

The built-in examples are demonstrations, not a pre-registered study. Before drawing conclusions, define hypotheses and outcomes in advance, use enough diverse prompts, track model/tokenizer versions, and test across multiple seeds or sampling settings where relevant.

## Quickstart

    python main.py

The original demo uses random weights to test the pipeline; use a pretrained model for useful experiments.

## License

MIT.
