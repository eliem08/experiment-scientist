# Experiment Scientist — Product Roadmap

## Product outcome

A reproducible experiment manager for LLM prompt and model research: define hypotheses, pre-register metrics, run paired conditions, analyze uncertainty, and publish a report tied to immutable configurations.

## Users and problem

- Applied AI researchers testing prompt strategies.
- Product teams comparing instructions or models.
- Students learning experimental design and avoiding cherry-picked examples.

## Current foundation

- Paired baseline/treatment runner, deterministic generation, word-count summaries, JSON/CSV output, dashboard and Colab notebook.
- **New:** validated JSON pre-registration spec, explicit outcome functions, model × seed × case matrix, per-observation CSV, spec SHA-256, paired case-level bootstrap intervals, and paired effect size.

## Roadmap

### Phase 1 — Scientific discipline
- [x] Move cases and hypotheses to a validated experiment-spec JSON file.
- [x] Add explicit outcome functions rather than relying only on word count.
- [ ] Store pinned model/tokenizer revisions and full package/environment versions.
- [x] Add a deterministic percentile-bootstrap interval for paired outcomes, with limitations documented.
- [x] Add repeated seeds and paired uncertainty analysis that aggregates within case before bootstrapping.
- [ ] Add a results report that distinguishes exploratory from pre-registered analysis.

### Phase 2 — Research workflow
- [ ] Dataset sampling and stratification.
- [x] Multiple model/condition matrix (cost and latency accounting remains open).
- [x] Statistical summaries and paired effect size with assumptions documented.
- [ ] Resume failed batches and cache identical requests.
- [ ] Export Markdown/HTML report and raw artifact bundle.

### Phase 3 — Final product

A notebook-friendly experiment lab with experiment templates, a reproducibility manifest, condition matrix, paired analysis, plots, and one-click report export.

## Success criteria

- A run can be reconstructed from its saved manifest.
- Hypotheses and primary metrics are stored before results.
- The interface warns about small samples, multiple comparisons, and exploratory analyses.
