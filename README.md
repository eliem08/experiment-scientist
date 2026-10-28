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
