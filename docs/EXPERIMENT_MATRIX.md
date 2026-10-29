# Pre-registered experiment matrix

The matrix runner complements `prompt_experiment.py` with a spec-driven model × seed × case experiment. It is intended for transparent, reproducible prompt ablations, not as an automatic judge of open-ended answer quality.

## 1. Define the experiment before inference

Copy `examples/uncertainty-ablation.spec.json` and edit:

- `title` and `hypothesis`: write the question and directional expectation in advance.
- `primary_metric`: reference one metric ID; this prevents quietly choosing a favorable outcome after seeing results.
- `metrics`: declare deterministic scoring rules and their intended direction.
- `cases`: give each prompt pair a stable, unique ID.

The runner rejects unknown metric types, duplicate IDs, invalid regex patterns, missing prompts, and a primary metric that is not declared. It hashes the exact spec file bytes before model inference and stores the SHA-256 in the result.

## 2. Run a matrix

\`\`\`bash
python experiment_matrix.py \
  --spec examples/uncertainty-ablation.spec.json \
  --model-id distilgpt2 \
  --model-id gpt2 \
  --seeds 17,29,43,71,101 \
  --device auto \
  --output outputs/uncertainty-ablation
\`\`\`

Use `--sample` for seeded sampling. Greedy decoding is the default. Each case is evaluated under both prompt conditions for each seed; the same seed is reset before each arm to support a paired comparison. For each model and metric, the summary first averages treatment-minus-baseline deltas within each case across seeds, then bootstraps those case-level deltas. This avoids falsely counting repeated seeds for the same prompt as independent prompt samples.

## 3. Interpret the outputs

- `matrix-results.json`: spec fingerprint, hypothesis, primary metric, model IDs, seed set, generation settings, summaries, all observations, and interpretation limits.
- `matrix-observations.csv`: one row per model × seed × case × metric, including scores, deltas, response hashes, and responses.

Each summary reports the mean paired delta, an exploratory percentile-bootstrap interval, standard deviation of case-level deltas, and paired Cohen's \(d_z\) when estimable. Positive deltas favor treatment only for metrics whose direction is `higher`; negative deltas favor treatment for `lower`. A `neutral` metric is diagnostic.

## 4. Current metric types

| Kind | Scoring rule | Caveat |
|---|---|---|
| `word_count` | Whitespace-separated token count | Length is not quality |
| `character_count` | Python string length | Length is not quality |
| `contains` | Case-insensitive substring presence (0/1) | Exact phrasing may miss paraphrases |
| `regex_match` | Whether the declared regular expression matches | Regex validity is not factual validity |
| `exact_match` | Trimmed exact string equality (0/1) | Appropriate only for tightly specified tasks |

The supported metrics are intentionally simple and auditable. Add a separately validated judge or task-specific evaluator only when you can define its reliability and failure modes; do not relabel heuristic scores as semantic correctness or safety.

## 5. Limitations and good practice

- The example has only two cases and is a tutorial fixture, not a publishable benchmark.
- Bootstrap intervals can be highly unstable with few cases. Increase diverse cases before interpreting intervals or effect sizes.
- Repeated seeds estimate generation variability under the chosen setup; they do not create additional independent task examples.
- The current runner stores the requested model IDs and generation settings, but it does not yet pin a Hub revision or capture full package/environment lockfiles. Pin model revisions and archive the environment for stricter replication.
- The spec hash supports auditability, but does not independently prove the spec was authored before any exploratory results.
- Results include model responses; protect the output directory if prompts or generations contain sensitive data.
- A passing rule-based metric or favorable effect size is not a safety or correctness certification.
