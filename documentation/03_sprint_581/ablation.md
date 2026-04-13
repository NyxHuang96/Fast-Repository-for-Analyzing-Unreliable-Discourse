---
title: "Ablation Study — Sprint 4 Detailed Analysis"
authors: "Marco Wang, Tianhao Cao, Yusen Huang, Darwin Zhang"
date: "2026-04-12"
course: "COLX 581 — Sprint 4"
disclaimer: "Documentation assisted by Claude Sonnet 4.6"
---

# Ablation Study — Sprint 4 Detailed Analysis

## Overview

By the end of Sprint 3 we had accumulated a diverse model zoo trained on the same 105-sample bilingual email dataset (Ham / Phish / Spam).

| Sprint | Model | Val Macro-F1 | Test Macro-F1 |
|:---|:---|:---:|:---:|
| S1 | TF-IDF + LinearSVC | 0.9221 | 0.6405 |
| S1 | DistilBERT (EN, fine-tuned) | 0.5714 | 0.5444 |
| S2 | TF-IDF + fastText (EN+ZH) + SVC | 0.8110 | **0.7566** |
| S2 | mDistilBERT full fine-tune | 0.6016 | 0.4703 |
| S2 | mDistilBERT frozen-3 layers | **0.8110** | 0.4101 |
| S2 | Motivated Ensemble (bias+entropy+grid) | 0.9221 | **0.8778** |
| S3 | DistilBERT MTL + Silver NER (λ=0.3) | 0.7231 | 0.4353 |
| S3 | TF-IDF + NER density + SVC | 0.8300 | — |

The Detailed Analysis section ablates **four design decisions we had not yet isolated**:

1. Whether fastText's gain is from *semantic geometry* or just *subword coverage*
2. How sensitive the Neural MTL model is to the NER auxiliary loss weight λ
3. Which of the three motivated ensemble layers drives its performance
4. Whether the Chinese fastText vectors add signal over English-only vectors

Each ablation has its own script in `src/04_sprint_581/` and produces a summary table comparing all configurations on the validation and test sets.

---

## Ablation 1 — Subword Robustness: fastText vs. Character-Level TF-IDF

**Script:** `src/04_sprint_581/ablation_1_char_tfidf.py`

### Research Question

The Sprint-2 TF-IDF + fastText model improved Test Macro-F1 from 0.6405 (Sprint-1) to 0.7566 (+0.12, +18% relative). fastText provides two things simultaneously:

- **Semantic geometry**: Words with related meanings sit nearby in the 300-d vector space ("phishing" and "fraud" are neighbouring vectors).
- **Subword coverage**: fastText uses character n-grams internally, so it produces meaningful vectors for OOV words and misspellings common in spam/phish emails (e.g., "v1agra", "ph1shing").

This ablation disentangles those two contributions by replacing fastText with a **character-level TF-IDF** (`analyzer='char_wb'`, n=(2,4)). Char-TF-IDF provides subword-level lexical patterns without any semantic geometry.

### Configurations

| Config | Description |
|:---|:---|
| A | Word TF-IDF only (Sprint-1 baseline, reproduced) |
| **B** | **Word TF-IDF + Character TF-IDF (this ablation)** |
| C | Word TF-IDF + fastText EN+ZH (Sprint-2 reference) |

### Design Details

- **Config B features:** A word-level TF-IDF (unigrams + bigrams, identical to Sprint-1) concatenated horizontally with a character-level TF-IDF (`char_wb`, n=(2,4), `sublinear_tf=True`). The `char_wb` mode pads token boundaries with whitespace so n-grams do not span word boundaries, which is appropriate for handling intentional misspellings in spam.
- **Why `sublinear_tf`?** Character n-gram features have very skewed frequency distributions — a few common sub-sequences appear extremely often. Log-scaling TF damps this effect and is standard practice for char n-gram models.
- All configs use `LinearSVC(class_weight='balanced', random_state=581)`.

### Interpretation of Results

**Expected finding:** If Config B Macro-F1 ≈ Config C, the fastText gain is mostly attributable to subword coverage, not semantic geometry. If Config B < Config C, semantic geometry (the continuous vector space) provides irreplaceable signal.

**Live results:**

| Configuration | Val Acc | Val Macro-F1 | Test Acc | Test Macro-F1 | Ham F1 | Phish F1 | Spam F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| A: Word TF-IDF only (S1 repro.) | 0.9167 | 0.9221 | 0.6875 | 0.7014 | 0.8889 | 0.6154 | 0.6000 |
| **B: Word TF-IDF + Char TF-IDF** | **0.8333** | **0.8110** | **0.7500** | **0.7556** | **1.0000** | **0.6667** | **0.6000** |
| C: Word TF-IDF + fastText (S2 repro.) | 0.8333 | 0.8110 | 0.7500 | 0.7566 | 0.8889 | 0.7143 | 0.6667 |

**Finding:** Config B (Char TF-IDF) achieves Test Macro-F1 = **0.7556**, nearly identical to fastText (0.7566). The difference is only +0.001 in favour of fastText. This strongly suggests that the Sprint-2 improvement was driven primarily by **subword lexical coverage**, not semantic geometry. The character n-gram features alone are sufficient to match the fastText improvement. Ham F1 is actually higher in Config B (1.000 vs 0.889) because char n-grams pick up character-level spam signals more directly, while fastText's soft semantic boundaries let one Ham mis-classify into Phish.

---

## Ablation 2 — NER Auxiliary Loss Weight λ in Neural MTL

**Script:** `src/04_sprint_581/ablation_2_ner_lambda.py`

### Research Question

Sprint-3 trained a DistilBERT multi-task model with the joint loss:

```
L = L_cls + λ * L_ner    (λ = 0.3)
```

The choice of λ=0.3 was motivated by the convention that the auxiliary task should be a regulariser, not dominate training. But is λ=0.3 actually the best setting? This ablation sweeps three values:

- **λ=0.0** — NER head is present but receives no gradient signal (pure single-task DistilBERT with the same architecture)
- **λ=0.3** — Sprint-3 setting (reproduced)
- **λ=1.0** — Equal-weight MTL (NER loss as strong as classification loss)

### Configurations

| Config | λ | Description |
|:---|:---:|:---|
| A | 0.0 | No NER supervision ("control" — MTL architecture but no auxiliary signal) |
| **B** | **0.3** | **Sprint-3 setting (reproduced)** |
| C | 1.0 | Equal-weight NER auxiliary loss |

### Design Details

All three configurations share:

- Same `distilbert-base-uncased` backbone, CLS head, and NER head architecture
- Same silver NER labels (spaCy `en_core_web_sm`)
- Same hyperparameters: lr=2e-5, batch=16, max_epochs=10, patience=3
- Same class-weighted CrossEntropyLoss for the classification head
- Same random seed (581)

The only variable is the λ coefficient multiplying `L_ner` in the joint loss.

**Note on λ=0.0:** This is not identical to the Sprint-1 baseline. The NER head forward pass still runs (it just contributes zero gradient). This makes it a true ablation control — same network, same compute graph, no auxiliary learning signal.

### Interpretation of Results

**Live results:**

| Configuration | Best Val Macro-F1 | Test Acc | Test Macro-F1 | Ham F1 | Phish F1 | Spam F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Sprint-1 DistilBERT (no MTL, documented) | 0.5714 | 0.6875 | 0.5444 | 0.8333 | 0.8000 | 0.0000 |
| **λ=0.0 (no NER signal, live)** | **0.8110** | **0.6875** | **0.6909** | **0.8000** | **0.7273** | **0.5455** |
| λ=0.3 (Sprint-3 repro., live) | 0.7231 | 0.5625 | 0.4353 | 0.6000 | 0.7059 | 0.0000 |
| λ=1.0 (equal weight, live) | 0.7231 | 0.5625 | 0.4444 | 0.6667 | 0.6667 | 0.0000 |

**Finding:** The most striking result is that **λ=0.0 (no MTL) dramatically outperforms both λ=0.3 and λ=1.0** in terms of Test Macro-F1 (0.6909 vs 0.4353 / 0.4444). This is the opposite of what the Sprint-3 hypothesis predicted. The key explanation is validation instability: with only 12 validation samples, the best checkpoint selected by early stopping is highly sensitive to random fluctuations. λ=0.0 achieved a best val F1 of **0.8110** by epoch 5 (the NER-free model converges faster and more stably), while λ=0.3 and λ=1.0 both peaked at 0.7231. The NER auxiliary loss makes the training loss landscape rougher — the model achieves decent validation F1 at epoch 3, but the checkpoint selected at that epoch happened to generalise poorly to the test set. Crucially, λ=0.0 is the only setting that achieved **non-zero Spam F1 (0.5455)** on the test set, confirming that the NER auxiliary task, at these two tested weights, does not help and may actually harm downstream Spam discrimination. The result further underscores the core finding from Sprint 3: the validation set (n=12) is too small for reliable model selection at this parameter scale.

---

## Ablation 3 — Motivated Ensemble Component Ablation

**Script:** `src/04_sprint_581/ablation_3_ensemble_components.py`

### Research Question

Sprint-2's motivated ensemble adds three layers to naive soft voting:

```
Raw probs
  → [Layer 1: Bias Correction]      per-class precision scaling + renorm
  → [Layer 2: Entropy Weighting]    uncertain models contribute less per-sample
  → [Layer 3: Grid-Search Weight]   data-driven global base weight from val set
  → Final prediction
```

These layers were motivated and designed together — but we never isolated which one actually drives performance. This ablation removes each layer individually and re-runs the full ensemble to measure the marginal contribution of each layer.

### Configurations

| Config | Bias Correction | Entropy Weighting | Grid Search | Description |
|:---|:---:|:---:|:---:|:---|
| **A** | ✓ | ✓ | ✓ | **Full motivated ensemble (baseline)** |
| B | ✗ | ✓ | ✓ | Bias correction removed |
| C | ✓ | ✗ | ✓ | Entropy weighting removed |
| D | ✓ | ✓ | ✗ | Fixed weight `w_bert=0.3` instead of grid search |

Config D uses `w_bert=0.3` (i.e., 0.7 weight on SVC) — the Sprint-2 manually-chosen best weight. This tests whether the grid search merely re-discovered that hand-picked value, or whether it found something better.

### Design Details

- **Shared inference:** BERT and SVC predictions are computed *once* from the same checkpoints and reused across all four configs. This ensures that differences in Macro-F1 are entirely due to the ablated ensemble layer, not any randomness in the underlying models.
- **Per-class reliabilities** are always computed from the validation set on BERT's and SVC's standalone predictions, even in Config B (which does not apply correction — reliabilities are computed but not used).
- The grid search in Configs A, B, C contains 21 candidate weights ∈ linspace(0, 1, 21) and selects the one that maximises validation macro-F1.

### Interpretation of Results

**Expected finding:**
- Removing bias correction (Config B) may hurt Spam F1 most, since bias correction specifically suppresses DistilBERT's over-prediction of Ham and partially recovers Spam probability mass.
- Removing entropy weighting (Config C) may have less impact if both models have similar calibration on most samples.
- Fixed weight vs. grid search (Config D vs A) measures whether the extra grid search complexity is justified or whether 0.3/0.7 is "good enough".

**Live results:**

| Configuration | Test Acc | Test Macro-F1 | Ham F1 | Phish F1 | Spam F1 |
|:---|:---:|:---:|:---:|:---:|:---:|
| Naive ensemble (0.5/0.5, documented) | 0.5625 | 0.5574 | 0.8889 | 0.5333 | 0.2500 |
| Naive best (0.3/0.7, documented) | 0.6250 | 0.6349 | 0.8889 | 0.5714 | 0.4444 |
| **A: Full motivated ensemble** | **0.8750** | **0.8778** | **1.0000** | **0.8000** | **0.8333** |
| B: No bias correction | 0.6250 | 0.6405 | 0.7500 | 0.5714 | 0.6000 |
| C: No entropy weighting | 0.8750 | 0.8778 | 1.0000 | 0.8000 | 0.8333 |
| D: Fixed w_bert=0.3 (no grid search) | 0.5625 | 0.4627 | 0.5882 | 0.8000 | 0.0000 |

**Finding:** The grid-searched weight is **the single most critical layer**. The full ensemble (A) found `w_bert=0.00` (i.e., SVC only after correction), achieving 0.8778 — a massive gain. Without grid search (Config D, fixed at 0.3/0.7), performance collapses to 0.4627 because the DistilBERT component actively poisons Spam predictions. Bias correction (Layer 1) is equally essential: removing it (Config B) drops performance to SVC standalone levels (0.6405), because uncorrected BERT probabilities suppress the corrected SVC Spam signal. Entropy weighting (Layer 2) is redundant here — Configs A and C produce identical results (0.8778), because after bias correction `w_bert=0.00` renders the entropy weights irrelevant (BERT gets zero base weight anyway).

---

## Ablation 4 — Bilingual fastText: Chinese vs. English Vectors

**Script:** `src/04_sprint_581/ablation_4_bilingual_fasttext.py`

### Research Question

Sprint-2 loaded **both English and Chinese fastText CC-100 vectors** and merged them into a single lookup table. The motivation was that our dataset is bilingual — each email has an English body (`text`) and a Chinese translation (`text_zh`). However, loading both doubles the memory usage and loading time.

Did the Chinese vectors actually contribute meaningful signal? Or was the improvement driven entirely by the English fastText embeddings?

### Configurations

| Config | fastText Vectors Used | Total Feature Dim |
|:---|:---|:---:|
| D | None (TF-IDF only — Sprint-1 baseline) | ~vocab |
| **A** | **EN only (cc.en.300.vec)** | ~vocab + 300 |
| B | ZH only (cc.zh.300.vec) | ~vocab + 300 |
| **C** | **EN + ZH (Sprint-2)** | ~vocab + 300 |

All four configurations share the same word-level TF-IDF features (unigrams + bigrams), the same bilingual tokeniser (jieba for Chinese, regex for English), and the same LinearSVC hyperparameters.

Note: Configs A, B, and C all produce a **300-d** embedding feature (mean-pooled over found tokens). The lookup table content differs, not the embedding dimensionality.

### Design Details

- **Mean pooling strategy:** For each email, tokens are looked up in the active embedding table (EN-only, ZH-only, or EN+ZH). Found tokens are averaged; OOV tokens contribute nothing. If no token is found, a zero vector is used (equivalent to missing embedding).
- **Config B (ZH only) on English text:** Tokens from the English body will almost always be OOV in the Chinese vocab table, so the embedding will be mostly zero-vectors. This tests whether ZH vectors can carry any signal even on emails that also have English bodies.
- **No re-tokenisation:** All configs use the same bilingual tokeniser. The difference is purely what is in the lookup table.

### Interpretation of Results

**Live results:**

| Configuration | Val Acc | Val Macro-F1 | Test Acc | Test Macro-F1 | Ham F1 | Phish F1 | Spam F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| D: TF-IDF only (Sprint-1 repro.) | 0.9167 | 0.9221 | 0.6875 | 0.7014 | 0.8889 | 0.6154 | 0.6000 |
| A: TF-IDF + EN fastText only | 0.8333 | 0.8110 | 0.7500 | 0.7381 | 1.0000 | 0.7143 | 0.5000 |
| **B: TF-IDF + ZH fastText only** | **0.8333** | **0.8110** | **0.8750** | **0.8778** | **1.0000** | **0.8333** | **0.8000** |
| C: TF-IDF + EN+ZH fastText (Sprint-2) | 0.8333 | 0.8110 | 0.7500 | 0.7566 | 0.8889 | 0.7143 | 0.6667 |

**Finding:** The results reveal a highly surprising pattern. All three fastText configurations produce identical **validation** F1 (0.8110), yet their **test** performance diverges substantially. Most strikingly, **ZH-only fastText (Config B) achieves the best test Macro-F1 of any configuration in this ablation: 0.8778** — even matching the Full Motivated Ensemble from Ablation 3. This is counter-intuitive: English tokens are almost entirely OOV in the Chinese vocabulary table, meaning most email embeddings are near-zero vectors. The zero-vector embedding still provides a constant 300-d signal to the SVC, which may be acting as a soft regulariser that penalises the SVC from over-weighting any single TF-IDF direction. EN-only (Config A, 0.7381) underperforms EN+ZH (Config C, 0.7566) by 0.02, suggesting that the Chinese vectors do add marginal Spam signal when combined — but the ZH-only result completely overturns the assumption that Chinese vectors are the weak partner. The test-set variance with n=16 is a key caution: a single misclassified sample swings Macro-F1 by ~0.04, so these results should not be over-interpreted. The consistent validation F1 across A, B, and C (all 0.8110) confirms that the vector differences are below the signal-to-noise threshold of the 12-sample validation set.

---

## How to Run

All ablation scripts use the same `data/processed/{train,validation,test}.jsonl` splits as the original sprint experiments.

```bash
# Ablation 1 — Char TF-IDF vs. fastText (no GPU needed)
python src/04_sprint_581/ablation_1_char_tfidf.py

# Ablation 2 — NER λ sweep (GPU recommended, ~10 min per λ value on CPU)
python -m spacy download en_core_web_sm   # if not already installed
python src/04_sprint_581/ablation_2_ner_lambda.py

# Ablation 3 — Ensemble components (requires models/baseline_neural/ checkpoint)
python src/04_sprint_581/ablation_3_ensemble_components.py

# Ablation 4 — Bilingual fastText (requires fastText .vec files)
python src/04_sprint_581/ablation_4_bilingual_fasttext.py
```

### Dependencies and Checkpoints

| Ablation | Requires |
|:---|:---|
| 1 | `jieba`, `scikit-learn` (standard project deps) |
| 2 | `transformers`, `torch`, `spacy`, `en_core_web_sm`, `src/03_sprint_581/silver_ner.py` |
| 3 | `transformers`, `torch`, `models/baseline_neural/` checkpoint from Sprint-1 |
| 4 | `jieba`, `scikit-learn`, `cc.en.300.vec` and/or `cc.zh.300.vec` in `models/fasttext/` |

---

## Consolidated Summary Table

### Test Set Results — All Ablations (Complete)

| Ablation | Configuration | Test Acc | Test Macro-F1 | Ham F1 | Phish F1 | Spam F1 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Reference** | Sprint-1: TF-IDF + SVC | 0.6250 | 0.6405 | 0.7500 | 0.5714 | 0.6000 |
| **Reference** | Sprint-2: TF-IDF + fastText + SVC | 0.7500 | 0.7566 | 0.8889 | 0.7143 | 0.6667 |
| **Reference** | Sprint-3: DistilBERT MTL λ=0.3 | 0.5625 | 0.4353 | 0.6000 | 0.7059 | 0.0000 |
| Ablation 1 | Word TF-IDF only (S1 repro.) | 0.6875 | 0.7014 | 0.8889 | 0.6154 | 0.6000 |
| Ablation 1 | Word TF-IDF + Char TF-IDF | 0.7500 | 0.7556 | 1.0000 | 0.6667 | 0.6000 |
| Ablation 1 | Word TF-IDF + fastText EN+ZH (S2 repro.) | 0.7500 | 0.7566 | 0.8889 | 0.7143 | 0.6667 |
| Ablation 2 | DistilBERT λ=0.0 (no NER signal) | 0.6875 | 0.6909 | 0.8000 | 0.7273 | 0.5455 |
| Ablation 2 | DistilBERT λ=0.3 (Sprint-3 repro.) | 0.5625 | 0.4353 | 0.6000 | 0.7059 | 0.0000 |
| Ablation 2 | DistilBERT λ=1.0 (equal weight) | 0.5625 | 0.4444 | 0.6667 | 0.6667 | 0.0000 |
| **Ablation 3** | **Full motivated ensemble** | **0.8750** | **0.8778** | **1.0000** | **0.8000** | **0.8333** |
| Ablation 3 | No bias correction | 0.6250 | 0.6405 | 0.7500 | 0.5714 | 0.6000 |
| **Ablation 3** | **No entropy weighting** | **0.8750** | **0.8778** | **1.0000** | **0.8000** | **0.8333** |
| Ablation 3 | Fixed weight 0.3/0.7 (no grid search) | 0.5625 | 0.4627 | 0.5882 | 0.8000 | 0.0000 |
| Ablation 4 | TF-IDF only (S1 repro.) | 0.6875 | 0.7014 | 0.8889 | 0.6154 | 0.6000 |
| Ablation 4 | TF-IDF + EN fastText only | 0.7500 | 0.7381 | 1.0000 | 0.7143 | 0.5000 |
| **Ablation 4** | **TF-IDF + ZH fastText only** | **0.8750** | **0.8778** | **1.0000** | **0.8333** | **0.8000** |
| Ablation 4 | TF-IDF + EN+ZH fastText (Sprint-2 repro.) | 0.7500 | 0.7566 | 0.8889 | 0.7143 | 0.6667 |

### Cross-Ablation Findings

| Finding | Key result |
|:---|:---|
| **Best single model across all ablations** | Full motivated ensemble (Abl. 3A) and ZH-only fastText (Abl. 4B) both hit **Macro-F1 = 0.8778** |
| **fastText gain is subword coverage, not semantics** | Char TF-IDF (0.7556) ≈ fastText (0.7566); ∆ = 0.001 |
| **NER auxiliary task hurts at both tested λ values** | λ=0.0 (no MTL) achieves 0.6909 vs 0.4353 for λ=0.3; only λ=0.0 produces non-zero Spam F1 |
| **Grid search is the critical ensemble layer** | Removing it (fixed 0.3/0.7) drops Macro-F1 from 0.8778 → 0.4627 |
| **Entropy weighting is redundant** | Configs A and C in Ablation 3 are identical (both 0.8778) |
| **ZH-only fastText outperforms EN+ZH combined** | 0.8778 vs 0.7566; likely a regularisation effect from near-zero embeddings |
