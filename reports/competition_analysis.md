# Deep Past Challenge: Public-Only Structured Analysis

Date: 2026-02-12  
Scope: Publicly accessible competition pages (`overview`, `data`, `models`, `discussion`, `leaderboard`, `rules`) and public thread content. `team` and `submissions` pages are not publicly rendered.

## 1) Task Summary

- Objective: build a model that translates **Old Assyrian Akkadian transliterations** into **English**.
- Training data:
  - `train.csv`: about 1.5k transliteration/translation pairs (document-level alignment).
- Test data:
  - visible `test.csv` is dummy; hidden test is used at scoring time (about 4k sentences from about 400 documents).
- Metric:
  - score is geometric mean of corpus-level BLEU and chrF++:
  - `Score = sqrt(BLEU * chrF++)`
  - sufficient statistics are aggregated corpus-wide (micro-averaged behavior).

## 2) Data Structure and Pitfalls

### Key files (publicly listed)

- `train.csv`
- `test.csv`
- `sample_submission.csv`
- `published_texts.csv`
- `publications.csv`
- `bibliography.csv`
- `OA_Lexicon_eBL.csv`
- `eBL_Dictionary.csv`
- `resources.csv`
- `Sentences_Oare_FirstWord_LinNum.csv`

### Practical pitfalls

- Hidden-test formatting is strict around `<gap>` / `<big_gap>`.
- Transliteration conventions vary (`sz`, `s,`, `u2`, subscript forms, braces/determinatives).
- Train/test structural mismatch: train mostly document-level, test scored sentence-level.
- OCR-derived supplemental text (`publications.csv`) is noisy and multilingual.
- Named entities (people/places/deities) are high-error and high-impact for BLEU/chrF.

### Recommended preprocessing

- Normalize transliteration to a canonical form:
  - ASCII variants -> canonical diacritics when relevant.
  - `x`/`[x]` -> `<gap>`, repeated/ellipsis breaks -> `<big_gap>`.
  - collapse mixed gap sequences deterministically.
  - standardize determinative forms in `{...}` while preserving semantics.
- Normalize whitespace/punctuation deterministically for both source and target.
- Preserve line-number semantics (`1`, `1'`, `1''`) as string tokens.

### Tokenization strategy

- Primary recommendation: **ByT5** (byte-level), robust to transliteration noise.
- Secondary: Flan-T5 / T5 variants with aggressive normalization and tighter length controls.

## 3) Discussion + Leaderboard + Models Insights

### Discussion themes (highest-signal public threads)

- `Compiled Discussions To Read (Avoid Bad Advice)` (71 votes)
- `Two practical stumbling blocks...` (61 votes): named entities + transliteration normalization dominate gains.
- `<gap>/<big_gap>` handling threads (40+ votes): host confirms hidden test follows these conventions.
- `Other Public Data` (26 votes): external public corpora can help pretraining/domain adaptation.
- `Starter code and some tips` (15 votes): task prefix and length/BP control discussed.

### Leaderboard snapshot (public)

- Public leaderboard uses about **34%** of test data; final ranking uses remaining **66%**.
- Top public scores observed in high-30s (e.g., ~39.2 at top at time of capture).
- High submission counts among top teams indicate iterative formatting + decoding optimization.

### Models page signal

- ByT5-family entries are frequent among strong public models.
- Public model scores show a wide spread, indicating preprocessing and decoding quality matter heavily.
- Non-ByT5 architectures can be competitive but are generally more sensitive to normalization/tokenization.

## 4) Rules Constraints Integrated into Strategy

- Code competition: notebook-based submissions.
- Runtime constraints: CPU/GPU notebook runtime limits; internet disabled at submit.
- Submission filename must be `submission.csv`.
- Team size max 5; daily submissions max 5; up to 2 final submissions.
- External data/tools allowed under accessibility/reasonableness criteria.

## 5) Model Comparison Table

| Track | Models | Speed | Memory | Complexity | Expected Score Band (public) | Risk |
|---|---|---:|---:|---:|---|---|
| Baseline | ByT5-small | Fast | Low-Med | Low | Mid/High-20s to low-30s (depends on preprocessing) | Lower |
| Top-LB | ByT5-base + ByT5-small + Flan-T5-base (staged) | Slower | Med-High | High | Higher ceiling; supports high-30s trajectory with strong preprocessing/ensemble | Higher |

## 6) Implementation Mapping

- `utils.py`
  - deterministic canonical normalization
  - exact competition metric wrapper
  - weak-pair mining hooks
  - grouped CV helpers and stress-slice tagging
  - ensemble consensus reranking
- `train.py`
  - config-driven baseline and staged multi-model training
  - stage-wise dataset composition (clean -> weak -> hard replay)
  - early stopping and checkpointing
- `inference.py`
  - batched generation
  - weighted vote or consensus rerank ensemble
  - deterministic postprocessing + schema validation

## 7) Sources (Public URLs)

- Overview: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/overview  
- Data: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/data  
- Models: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/models  
- Discussion (votes): https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion?sort=votes  
- Leaderboard: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/leaderboard  
- Rules: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/rules  
- Metric notebook reference: https://www.kaggle.com/code/metric/dpi-bleu-chrf  
- Key discussion thread IDs:
  - https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/665209
  - https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/664411
  - https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/664518
  - https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/668402
  - https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/663357

## 8) Assumptions

- `team` and `submissions` pages are not publicly available without authenticated context.
- `code` listing details are partially constrained by visibility/authentication.
- Strategy is intentionally public-only and reproducible under those constraints.
