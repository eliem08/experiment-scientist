# Experiment Scientist — Product Roadmap

## Product outcome

A reproducible experiment manager for LLM prompt and model research: define hypotheses, pre-register metrics, run paired conditions, analyze uncertainty, and publish a report tied to immutable configurations.

## Users and problem

- Applied AI researchers testing prompt strategies.
- Product teams comparing instructions or models.
- Students learning experimental design and avoiding cherry-picked examples.

## Current foundation

- Paired baseline/treatment runner, deterministic generation, word-count summaries, JSON/CSV output, dashboard and Colab notebook.

## Roadmap

### Phase 1 — Scientific discipline
- [ ] Move cases and hypotheses to a validated experiment-spec YAML/JSON file.
- [ ] Add explicit outcome functions rather than relying only on word count.
- [ ] Store exact model/tokenizer revision, prompt hashes, generation config and package versions.
- [x] Add a deterministic percentile-bootstrap interval for word-count deltas (clearly marked exploratory for the current tiny sample).\n- [ ] Add repeated seeds and stronger paired uncertainty analysis.
- [ ] Add a results report that distinguishes exploratory from pre-registered analysis.

### Phase 2 — Research workflow
- [ ] Dataset sampling and stratification.
- [ ] Multiple model/condition matrix with cost and latency accounting.
- [ ] Statistical summaries and effect sizes with assumptions documented.
- [ ] Resume failed batches and cache identical requests.
- [ ] Export Markdown/HTML report and raw artifact bundle.

### Phase 3 — Final product
A notebook-friendly experiment lab with experiment templates, a reproducibility manifest, condition matrix, paired analysis, plots, and one-click report export.

## Success criteria

- A run can be reconstructed from its saved manifest.
- Hypotheses and primary metrics are stored before results.
- The interface warns about small samples, multiple comparisons, and exploratory analyses.
