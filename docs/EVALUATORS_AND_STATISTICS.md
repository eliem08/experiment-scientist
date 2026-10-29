# Evaluators and paired analysis

This guide documents the additive research-core APIs. Existing CLI commands and artifact schemas remain unchanged.

## Deterministic evaluators

`evaluation_core.py` exposes a small structural `Evaluator` protocol and versioned `EvaluationResult` records.

```python
from evaluation_core import ContainsEvaluator, EvaluationInput, ExactMatchEvaluator

item = EvaluationInput(case_id="case-1", response="42", reference="42")
result = ExactMatchEvaluator().evaluate(item)
assert result.score == 1.0

uncertainty = ContainsEvaluator("I don't know").evaluate(
    EvaluationInput(case_id="case-2", response="I don't know the answer.")
)
assert uncertainty.score == 1.0
```

Built-ins include exact match, substring presence, regular-expression match, and selected JSON-field equality. They are intentionally transparent; substring or regex matches are not semantic correctness.

## Provider-neutral LLM judge

The judge adapter requires an injected callable and does not make network calls or manage credentials.

```python
from evaluation_core import EvaluationInput, LLMJudgeEvaluator

def local_or_provider_adapter(prompt, context):
    # Call a model using your own adapter and return a JSON string or mapping.
    return {"score": 0.8, "label": "good", "rationale": "Satisfies the rubric."}

judge = LLMJudgeEvaluator(
    judge_id="my-judge",
    judge_revision="immutable-model-revision",
    rubric="Assess whether the response follows the declared instruction.",
    judge_call=local_or_provider_adapter,
    max_retries=2,
)
result = judge.evaluate(EvaluationInput("case-1", "Candidate response"))
```

The judge must return `score`, `label`, and `rationale`. Scores outside the declared range, malformed responses, and adapter failures produce an explicit failed result with `score=None`; failures must not be treated as zero. The output retains the raw decision and a hash of the normalized decision for auditability. Raw judge decisions may contain sensitive text, so apply your own retention policy.

An LLM judge is not ground truth. Calibrate it against human annotations, inspect disagreement, test ordering and verbosity effects, and record judge prompt/model revisions before relying on its scores.

## Paired statistical analysis

`statistical_analysis.py` provides independent-unit aggregation, percentile bootstrap intervals, paired-delta effect size, a sign-flip permutation test, and Holm adjustment.

```python
from statistical_analysis import paired_unit_deltas, mean_confidence_interval, sign_flip_permutation_test

rows = [
    {"case_id": "a", "baseline_score": 0.0, "treatment_score": 1.0},
    {"case_id": "a", "baseline_score": 0.5, "treatment_score": 1.0},
    {"case_id": "b", "baseline_score": 0.0, "treatment_score": 0.0},
]
pairs = paired_unit_deltas(rows)
deltas = [row["delta"] for row in pairs]
interval = mean_confidence_interval(deltas, seed=17)
permutation = sign_flip_permutation_test(deltas, seed=17)
```

Repeated rows sharing a `case_id` are averaged within each arm, yielding one delta per independent case. Change the pairing key only when the experimental design justifies another independent unit. The bootstrap interval is a percentile interval and can be unstable for small datasets. The sign-flip test assumes sign exchangeability under the null; do not use it automatically when the design does not support that assumption.

Holm adjustment controls family-wise error across a declared family of p-values. Define the family before examining results. These utilities are not an automatic replacement for a registered analysis plan or expert statistical review.
