---
title: "Motivated Ensemble: Bias Correction, Entropy Weighting, and Validation-Tuned Global Weights"
author: Tianhao Cao
date: "2026-04-03"
Disclaimer: This document is generated with the help of Claude Sonnet 4.6
---

## Overview

This document describes `motivated_ensembling.py`, an improved ensemble strategy that addresses the three root causes of failure identified in the [naive soft-voting baseline](ensemble_baseline.md):

1. **Model bias** — DistilBERT over-predicts Ham and near-never predicts Spam; the SVC has its own per-class precision imbalances.
2. **Calibration mismatch** — DistilBERT produces overconfident (peaky) softmax distributions; the SVC's softmax'd decision-function scores are more diffuse. Equal weighting does not mean equal influence.
3. **Fixed global weights** — The baseline tried three hand-picked weight ratios. There is no principled reason to prefer 0.3/0.7 over 0.35/0.65; the optimal split should be derived from data.

The motivated ensemble wraps the original soft-voting step in three layers that each address one of these problems.

---

## Why This Combination?

The three improvements are ordered so that each layer operates on already-cleaned input from the previous one:

```
Raw probs
  → [Layer 1: Bias Correction]      remove class-level distortions
  → [Layer 2: Entropy Weighting]    reduce influence of uncertain predictions
  → [Layer 3: Global Weight Tuning] find the best base ratio from data
  → Final prediction
```

Applying bias correction *before* computing entropy matters because a distorted probability distribution (e.g., near-zero Spam probability in DistilBERT) would otherwise produce misleadingly low entropy — it looks confident when it is simply biased. Correcting biases first makes the entropy signal more informative.

Applying entropy weighting *before* the global weight search means that the grid search optimises over the full motivated ensemble pipeline, not just over a raw average. The best base weight found this way is the one that works best in combination with the other two layers, not in isolation.

---

## Component 1 — Per-Class Bias Correction

### What it does

For each model, a **reliability score** is computed per class on the validation set. Reliability is defined as the model's **precision** for that class: out of all samples the model predicted as class *c*, what fraction were actually class *c*?

```
reliability[c] = TP_c / (TP_c + FP_c)
```

Before combining the two models' probabilities, each model's probability for class *c* is multiplied by its reliability for class *c*. The resulting vector is renormalised to sum to 1.

```python
corrected_probs = raw_probs * reliabilities   # element-wise broadcast
corrected_probs /= corrected_probs.sum(axis=1, keepdims=True)
```

### Why precision specifically?

Precision captures over-prediction. A model with low Spam precision is over-calling Spam — it assigns high probability to Spam even on non-Spam samples. Scaling down by precision reduces exactly this systematic overconfidence.

Recall was not used because it captures *under*-prediction (missed positives), which would require scaling *up* rather than scaling down. Scaling up an already uncertain probability can amplify noise. The conservative approach — only suppressing over-confident classes — is more stable with small datasets.

### What this fixes

The baseline doc noted that DistilBERT's Spam probability was near-zero for every sample, meaning it never over-predicted Spam. Its reliability problem is the opposite: it over-predicts Ham. Bias correction suppresses DistilBERT's Ham probability and partially redistributes probability mass to Phish and Spam, where the SVC is stronger.

---

## Component 2 — Entropy-Based Per-Sample Confidence Weighting

### What it does

After bias correction, the **Shannon entropy** of each model's corrected probability distribution is computed per sample:

```
H(p) = -Σ p_c * log(p_c)
```

This is normalised by the maximum possible entropy `log(n_classes)` and inverted to produce a **confidence score** in [0, 1]:

```
confidence = 1 - H(p) / log(n_classes)
```

A uniform distribution (total uncertainty) gives confidence = 0. A one-hot distribution (total certainty) gives confidence = 1.

The per-sample effective weight for each model is then:

```
eff_weight_bert_i = base_w_bert * confidence_bert_i
eff_weight_svc_i  = base_w_svc  * confidence_svc_i
```

These are normalised to sum to 1 per sample before computing the weighted average.

### Why entropy?

The key insight is that **the global weight only tells us which model is better on average, but not on this specific sample**. A model that is globally weaker may still be more reliable on individual samples where it happens to produce a confident prediction.

Entropy is a natural measure of a model's uncertainty about its own output, computed directly from the probabilities it already produces — no additional data or labels are needed at test time.

### What this fixes

The baseline's calibration mismatch problem: DistilBERT's overconfident (peaky) distributions made it dominate soft voting even with reduced global weight. With entropy weighting, DistilBERT's confident predictions receive high weight but its genuinely uncertain predictions (e.g., on Spam samples it has never learned) contribute very little, letting the SVC steer those decisions.

---

## Component 3 — Validation-Tuned Global Base Weight

### What it does

A grid search is run over 21 candidate values of `base_w_bert` ∈ {0.0, 0.05, 0.10, …, 1.0}. For each candidate, the full motivated ensemble pipeline (bias correction + entropy weighting + weighted combination) is applied to the **validation set**, and macro-F1 is computed. The candidate that maximises validation macro-F1 is selected as the base weight for final test evaluation.

```python
for w_bert in linspace(0.0, 1.0, 21):
    preds = motivated_ensemble_core(val_probs, ..., base_w_bert=w_bert)
    f1    = macro_f1(val_labels, preds)
    if f1 > best_f1:
        best_w_bert = w_bert
```

### Why grid search over the validation set?

The baseline tried three hand-picked ratios and found that heavier SVC weight was better, but could not determine *how much* heavier was optimal. A grid search replaces intuition with measurement.

Using macro-F1 (rather than accuracy) as the search objective reflects the class imbalance in the dataset — Spam samples are the minority and the hardest to classify, but they matter as much as Ham and Phish in the final score.

### Why a coarse grid (21 steps) rather than finer resolution?

With 16 test samples, the optimal weight boundary is unlikely to lie at a precision finer than 0.05. A finer grid on a small dataset risks over-fitting the validation set noise. The grid is intentionally kept coarse to retain generalisability.

---

## Full Pipeline Summary

| Step | Operation | Input | Output |
|------|-----------|-------|--------|
| 1 | Compute per-class reliability | Val predictions vs. val labels | `bert_rel[3]`, `svc_rel[3]` |
| 2 | Grid search base weight | Val probs + reliabilities | `opt_w_bert` ∈ [0, 1] |
| 3 | Apply bias correction (test) | Raw test probs | Corrected test probs |
| 4 | Compute entropy confidence (test) | Corrected test probs | Per-sample weights |
| 5 | Weighted combination (test) | Corrected probs + per-sample weights | Combined distribution |
| 6 | Argmax | Combined distribution | Final prediction |

Steps 1 and 2 use only the validation set and are computed once. Steps 3–6 are applied to every test sample independently.

---

## Comparison to Baseline

The naive ensemble averaged raw probabilities with a fixed weight. The motivated ensemble differs in three ways:

| Property | Naive Ensemble | Motivated Ensemble |
|----------|---------------|-------------------|
| Class bias | Untouched | Corrected via per-class precision |
| Sample-level uncertainty | Ignored | Modulated via entropy |
| Weight selection | Hand-picked (3 configs tested) | Data-driven (grid search on val set) |
| Weight granularity | Global (same for every sample) | Per-sample (varies by confidence) |

The naive ensemble's best result (Macro-F1 = 0.6349, weights 0.3/0.7) serves as the direct comparison point. The motivated ensemble should be evaluated against this figure to assess whether the additional complexity is justified.

---

## Limitations

- **Small validation set.** The reliability scores and grid-searched weight are computed on a validation set of limited size. Per-class precision estimates may be noisy, especially for Phish and Spam which have fewer samples. The reliabilities should be interpreted as rough corrections rather than precise calibration factors.
- **SVC is re-trained at runtime.** Unlike the DistilBERT model which is loaded from a saved checkpoint, the SVC is trained from scratch every run. Results are deterministic (fixed random seeds) but runtime varies with dataset size.
- **Entropy is computed on biased-corrected probs.** If bias correction produces a near-uniform distribution (because reliabilities are very low), the entropy signal becomes uninformative. This is an edge case unlikely to occur with reasonable model performance.
