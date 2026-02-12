# Deep Past MT Pipeline (Akkadian -> English)

This repository implements a reproducible, Kaggle-ready machine translation pipeline for the competition:

- **Deep Past Challenge - Translate Akkadian to English**
- Format: **Code Competition**
- Submission artifact: **`submission.csv`**

It provides:

- A fast strong baseline (`google/byt5-small`)
- A staged top-LB pipeline (ByT5-base + ByT5-small + Flan-T5-base)
- Deterministic preprocessing and metric parity utilities
- CV-friendly training and ensemble inference

## Repository Layout

- `train.py`: train baseline or top-LB staged models from YAML config
- `inference.py`: generate predictions + ensemble + write `submission.csv`
- `utils.py`: normalization, metric, weak-data mining, CV helpers, reranking
- `configs/baseline.yaml`: fast baseline track
- `configs/top_lb.yaml`: top-LB staged multi-model track
- `reports/competition_analysis.md`: structured public-only competition analysis

## Environment

Recommended packages (Kaggle notebook already has most):

```bash
pip install -r requirements.txt
```

## Quick Start

### 1) Run utility fixtures (metric + preprocessing checks)

```bash
python utils.py --self-check
```

### 2) Train fast baseline

```bash
python train.py --config configs/baseline.yaml
```

### 3) Inference baseline

```bash
python inference.py --config configs/baseline.yaml --summary /kaggle/working/outputs/baseline/training_summary.json
```

### 4) Train top-LB staged pipeline

```bash
python train.py --config configs/top_lb.yaml
```

### 5) Inference top-LB ensemble

```bash
python inference.py --config configs/top_lb.yaml --summary /kaggle/working/outputs/top_lb/training_summary.json
```

Final file will be written to:

- `/kaggle/working/outputs/<track>/submission.csv`

## Smoke Test

For a fast end-to-end check (50 rows):

```bash
python train.py --config configs/baseline.yaml --smoke
python inference.py --config configs/baseline.yaml --summary /kaggle/working/outputs/baseline/training_summary.json
```

## Reproducibility and Validation

- Seeded training (`global.seed` in config)
- Grouped fold split by `group_id` (`5` folds by default; `val_fold=0` pseudo-public)
- Validation metric: `sqrt(BLEU * chrF++)` via SacreBLEU at corpus-level (micro)
- Submission schema and row-order validation enforced before writing CSV

## Notes on Data and Rules

- Hidden test differs from dummy `test.csv`; score is computed only during submission run.
- External public data is allowed by competition rules, subject to accessibility/reasonableness.
- Competition notebooks must run with internet disabled and within runtime limits.

## Typical Runtime Guidance

- Baseline (`byt5-small`): fastest, low memory, strong starting public score
- Top-LB (`byt5-base` + ensemble): higher memory/time, better ceiling

## Common Failure Modes

- Unicode and transliteration mismatch (especially `sz/s,` forms and gap markers)
- Overfitting public leaderboard through brittle formatting hacks
- Invalid submission ordering (`id` mismatch)

Use `utils.py --self-check` and review normalized training samples before long runs.
