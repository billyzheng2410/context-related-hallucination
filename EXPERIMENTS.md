# Local Experiment Notes

This repo now has a small reproducible path for turning the hand-built
`better dataset/data.json` file into model-ready counterfactual samples.

## Environment

Use the virtualenv interpreter directly:

```bash
hallucination_env/bin/python --version
```

In this workspace, `source hallucination_env/bin/activate` did not reliably
put the virtualenv `python` first on `PATH`, so the scripts and config use
`hallucination_env/bin/python` explicitly.

## Prepare The Better Dataset

Generate normalized counterfactual rows:

```bash
hallucination_env/bin/python scripts/prepare_better_dataset.py
```

Generate rows filtered to answers that are single tokens under Pythia-410M:

```bash
hallucination_env/bin/python scripts/prepare_better_dataset.py \
  --model-name EleutherAI/pythia-410m \
  --local-files-only \
  --filter-single-token-answers
```

Outputs:

- `data/processed/better_counterfactual.jsonl`
- `data/processed/better_counterfactual_summary.json`

Current Pythia-filtered output:

- 1598 usable rows
- 97 skipped multi-token answer rows
- 6 relation families: taxonomy, partwhole, synonym, antonym, causation, spatial

The preparation script also avoids `gold == distractor` examples and fixes
simple article mismatches such as `A eagle` -> `An eagle`.

## Margin Baseline Smoke Test

Run a cheap next-token margin baseline before any Gumbel search:

```bash
hallucination_env/bin/python scripts/run_margin_baseline.py \
  --local-files-only \
  --offline \
  --limit 30
```

Outputs:

- `results/baselines/pythia_410m_margin_results.jsonl`
- `results/baselines/pythia_410m_margin_summary.json`

The current 30-row smoke test on `EleutherAI/pythia-410m` completed on CPU.
It produced a lower attacked margin than clean margin, which is a useful sign
that the counterfactual context can perturb next-token predictions.

## Gumbel Head Search

Run a tiny smoke test:

```bash
hallucination_env/bin/python scripts/run_gumbel_head_search.py \
  --local-files-only \
  --offline \
  --limit 4 \
  --eval-limit 4 \
  --epochs 2 \
  --batch-size 2 \
  --eval-batch-size 2 \
  --top-k-heads 2 \
  --dtype float32
```

Run a more useful small taxonomy experiment:

```bash
hallucination_env/bin/python scripts/run_gumbel_head_search.py \
  --local-files-only \
  --offline \
  --relation taxonomy \
  --domain natural \
  --template-idx 0 \
  --limit 24 \
  --eval-limit 24 \
  --epochs 30 \
  --batch-size 4 \
  --eval-batch-size 8 \
  --top-k-heads 10 \
  --dtype float32
```

Outputs are written under:

- `results/gumbel/<run_name>/summary.json`
- `results/gumbel/<run_name>/history.json`
- `results/gumbel/<run_name>/base_eval.jsonl`
- `results/gumbel/<run_name>/learned_ablation_eval.jsonl`
- `results/gumbel/<run_name>/random_ablation_eval.jsonl`

The Gumbel script now directly consumes `data/processed/better_counterfactual.jsonl`.
It also includes a random-head ablation baseline, so the first real comparison
is no longer just "base vs learned ablation" but "learned ablation vs random
ablation with the same number of heads".

Add bootstrap confidence intervals for an existing run:

```bash
hallucination_env/bin/python scripts/summarize_ablation_run.py \
  results/gumbel/EleutherAI_pythia-410m_taxonomy_natural_seed42 \
  --n-bootstrap 1000
```

This writes:

- `results/gumbel/<run_name>/summary_with_ci.json`

Current smoke-test output confirms that Pythia loading, attention hooks, mask
training, learned-head ablation, random-head ablation, and result writing all
work locally. The tiny smoke test is only a pipeline check; do not interpret
its metrics as a research result.

## Next Step

The next implementation step should run relation/domain grids with repeated
seeds, then aggregate `summary_with_ci.json` files into paper-style tables.
