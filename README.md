# Experiment Scientist

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eliem08/experiment-scientist/blob/main/colab.ipynb)


Run a bounded, reproducible experiment plan with baseline, treatment, and control prompts.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-GPU%20or%20CPU-ee4c2c)

## Quickstart

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Default run uses a tiny randomly initialized model to validate the software path and does not claim useful model behavior.

## Use a pretrained model

Set `NEURONLAB_MODEL` or pass `--real-model --model-id MODEL_ID`. For example, select a small causal language model you are licensed to use:

```bash
python main.py --real-model --model-id distilgpt2 --text "The sky is"
```

Model weights take disk, RAM, and often VRAM. The runtime detects CUDA, Apple MPS, or CPU automatically. Install the appropriate PyTorch build for your hardware if needed.

## Google Colab

Open `colab.ipynb` in Colab, select **Runtime → Change runtime type → T4 GPU** when available, then run all cells. The same notebook also runs on CPU, but inference is slower. Free GPU availability and quotas vary.

## Reproducibility and limitations

- Save output artifacts under `outputs/`. Repeat experiments on held-out inputs before making claims.
- Random demo weights only exercise code paths. They are not suitable for quality, safety, or interpretability conclusions.
- Do not treat lexical heuristics or activation patterns as reliable evidence of intent or deception.
- Activation steering may harm other capabilities; record baselines and random controls.
- Do not expose model inference to the public internet without authentication and resource controls.

## License

MIT.
