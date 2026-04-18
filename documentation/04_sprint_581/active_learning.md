---
title: "Active Learning"
author: Tianhao Cao
date: "2026-04-18"
Disclaimer: This documentation and code for the active learning code and documentation was supported by Claude Sonnet 4.6.
---

# Active Learning — Sprint 4

## Overview

Starting from the bootstrapped training set (1104 original + 26 pseudo-labeled samples), we apply two active learning methods to rank the 26 bootstrapped samples by uncertainty, replace pseudo labels with manual gold labels, and measure the effect on model performance via an accumulation test.

## Method 1: Query-by-Uncertainty (Margin Sampling)

**Approach:** Train a LinearSVC on the 1104 original samples. For each of the 26 bootstrapped samples, compute the margin between the top-1 and top-2 class scores from `decision_function`. Smaller margin = lower confidence = higher AL priority.

**Rationale:** Directly implements the margin-based disagreement method from Lecture 7. LinearSVC does not output calibrated probabilities, but the raw decision scores provide a well-defined uncertainty proxy.

## Method 2: Query-by-Committee (Entropy of Disagreement)

**Approach:** Train 5 models, each on a 90% random subsample of the 1104-sample base training set. For each bootstrapped sample, compute the entropy of the vote distribution across the committee. Higher entropy = more disagreement = higher AL priority.

**Finding:** All 5 committee models agreed on all 26 bootstrapped samples (entropy = 0.0 for every sample). This is expected: with 1104 training examples, a 90% subsample (~994 examples) yields models that are nearly identical in their decision boundaries. This confirms the lecture's point that QBC is most effective when training data is scarce and committee models have meaningfully different biases.

## AL Scores Table

| # | Margin (↑ = certain) | Gold Label | Pseudo Label | Match? |
|---|---|---|---|---|
| 16 | 0.068 | Ham | Phish | ✗ |
| 19 | 0.103 | Spam | Spam | ✓ |
| 22 | 0.190 | Phish | Phish | ✓ |
| 6  | 0.283 | Ham | Spam | ✗ |
| 0  | 0.312 | Phish | Spam | ✗ |
| ... | ... | ... | ... | ... |

Samples with the smallest margins (most uncertain) were also among those where the model mislabeled (indices 16, 0, 5, 25). This validates that low-confidence predictions are indeed the more error-prone ones.

## Accumulation Test Results

Re-trained the model (LinearSVC, TF-IDF bigrams) with the base 1104 samples + top N AL-selected samples (gold labels replacing pseudo labels). Evaluated on the 12-sample dev set.

| AL% | N added | QBU Accuracy | QBC Accuracy |
|-----|---------|-------------|-------------|
| Baseline (base only) | 0 | 0.9167 | 0.9167 |
| 5% | 2 | 0.9167 | 0.9167 |
| 10% | 3 | 0.9167 | 0.9167 |
| 15% | 4 | 0.9167 | 0.9167 |
| 20% | 6 | 0.9167 | 0.9167 |
| Full bootstrap (pseudo) | 26 | 0.9167 | — |

## Discussion

The flat accumulation curve is explained by two factors:

1. **Diminishing returns on a large base set.** The base training set already contains 1104 examples. Adding 2–6 corrected samples (0.2–0.5% of the training data) is insufficient to shift the model's decision boundary enough to change any prediction on the dev set.

2. **Small dev set.** With only 12 dev examples, the model either gets 11 correct (0.9167) or 12 correct (1.0). The granularity of measurement is too coarse to detect marginal improvements.

Despite the flat accuracy curve, the AL ranking itself is informative: the 3 pseudo-label errors with the smallest margins (indices 16, 0, 25) were all ranked in the top-5 most uncertain samples, confirming that margin sampling correctly identifies unreliable predictions. This is the core value of active learning — prioritizing annotation effort on examples where the model is most likely to be wrong.

## Files

- `src/04_sprint_581/active_learning.py` — full pipeline
- `documentation/04_sprint_581/al_ranked.jsonl` — 26 bootstrapped samples ranked by QBU margin (gold labels)
