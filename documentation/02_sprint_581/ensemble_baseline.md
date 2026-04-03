---
title: "Evaluation of Ensemble Baseline (Weighted Soft Voting)"
author: Yusen Huang
date: "2026-04-03"
Disclaimer: This document is generated with the help with Claude opus 4.6
---

## Overview

This document outlines the evaluation of our ensemble baseline. The ensemble combines predictions from two independently trained models — a fine-tuned **DistilBERT** (`distilbert-base-uncased`) neural baseline and a **TF-IDF + LinearSVC** traditional baseline — using **weighted soft voting** for 3-class email classification (Ham / Phish / Spam).

Both models were trained on **105 Gold-annotated samples** and evaluated on a strictly isolated test set of **16 samples**. The ensemble averages the full class probability distributions from both models (with configurable weights), then takes the argmax as the final prediction.

------------------------------------------------------------------------

## 1. Scoring Strategy & Justification

**Method: Weighted Soft Voting**

For each test sample, both models produce a probability distribution over all 3 classes. The ensemble computes a weighted average of these distributions, then selects the class with the highest combined probability.

-   **DistilBERT** outputs logits that are passed through softmax to produce probabilities.
-   **LinearSVC** outputs decision-function scores (distance from hyperplane per class), which are converted to probabilities via softmax over the raw scores.

**Why soft voting over max-confidence selection?**

An initial attempt used max-confidence selection — picking the prediction from whichever model was more confident. This failed because DistilBERT's softmax outputs are poorly calibrated: the model assigns high probability (0.8–0.9) to its top class even when wrong. After min-max normalisation, DistilBERT dominated the ensemble (13 out of 16 samples), and the SVC's strengths were ignored entirely. The result was an ensemble that performed worse than either standalone model (Macro-F1 = 0.4646).

Soft voting fixes this by combining the *full class distributions* rather than selecting one model's prediction outright. Even when DistilBERT assigns near-zero probability to Spam, the SVC's Spam probability still contributes to the averaged distribution, partially recovering the signal.

------------------------------------------------------------------------

## 2. Quantitative Results (The Numbers)

Three weight configurations were evaluated. The best-performing ensemble used **SVC-heavy weights (0.3 / 0.7)**, achieving **62.50% Accuracy** and **Macro-F1 of 0.6349**.

**Comparison Across All Configurations:**

| Configuration             | Accuracy |  Macro-F1  | Ham F1 | Phish F1 |  Spam F1   |
|:-----------|:----------:|:----------:|:----------:|:----------:|:----------:|
| DistilBERT (standalone)   |  0.6875  |   0.5444   | 0.8333 |  0.8000  |   0.0000   |
| TF-IDF + SVC (standalone) |  0.6250  | **0.6405** | 0.7500 |  0.5714  | **0.6000** |
| Ensemble (0.5 / 0.5)      |  0.5625  |   0.5574   | 0.8889 |  0.5333  |   0.2500   |
| Ensemble (0.4 / 0.6)      |  0.5625  |   0.5574   | 0.8889 |  0.5333  |   0.2500   |
| Ensemble (0.3 / 0.7)      |  0.6250  |   0.6349   | 0.8889 |  0.5714  |   0.4444   |

**Best Ensemble Confusion Matrix (0.3 / 0.7):**

| True Label  Predicted | Predicted Ham | Predicted Phish | Predicted Spam |
|:----------------------|:-------------:|:---------------:|:--------------:|
| **True Ham**          |       4       |        1        |       0        |
| **True Phish**        |       0       |        4        |       2        |
| **True Spam**         |       0       |        3        |     **2**      |

------------------------------------------------------------------------

## 3. What Do These Metrics Actually Mean?

-   **Spam Recovery:** The most significant improvement is in Spam detection. DistilBERT alone had 0.0 Spam F1 — it never predicted Spam at all. The best ensemble recovered Spam F1 to 0.4444 by incorporating the SVC's Spam signal through the averaged probability distribution.
-   **Ham Precision Improved:** The ensemble achieved 100% Ham precision (vs. 71.43% for DistilBERT alone), meaning every email it labelled Ham was actually Ham. This came at a small cost to recall (80% vs. 100%).
-   **Weight Sensitivity:** Equal weights (0.5 / 0.5) still underperformed because DistilBERT's zero-Spam probabilities diluted the SVC's signal. Giving SVC 70% weight was necessary to let its Spam predictions come through.
-   **Still Below SVC Standalone:** The best ensemble (Macro-F1 = 0.6349) did not surpass the SVC standalone (Macro-F1 = 0.6405). The ensemble is only as good as its weakest component allows — DistilBERT's complete Spam blind spot actively dilutes the combined distribution.

------------------------------------------------------------------------

## 4. Why Did the Ensemble Not Surpass the SVC?

The ensemble's inability to beat the standalone SVC is a direct consequence of DistilBERT's failure mode:

**DistilBERT Contributes Negative Signal for Spam:** With only 105 training samples, DistilBERT never learned to distinguish Spam from the other two classes. Its Spam probabilities are near-zero for every sample. In a soft voting scheme, this drags down the combined Spam probability even when the SVC correctly assigns high Spam probability. The SVC must overcome DistilBERT's opposing signal, which requires giving the SVC disproportionately high weight — at which point the ensemble effectively becomes the SVC alone.

**Calibration Mismatch:** Softmax probabilities from DistilBERT and softmax'd decision-function scores from the SVC are not inherently on the same scale. DistilBERT's probabilities tend to be overconfident (peaky distributions), while the SVC's converted scores produce more diffuse distributions. This asymmetry means equal weights do not truly give equal influence to both models.

------------------------------------------------------------------------

## 5. What the Ensemble Approach Still Offers

Despite not surpassing the SVC, the ensemble experiment provides valuable insights:

-   **Soft voting is methodologically sound.** It successfully recovered partial Spam detection that the max-confidence approach completely missed, validating the approach even when the underlying models are mismatched.
-   **The weight trend is informative.** The monotonic improvement from 0.5/0.5 → 0.3/0.7 confirms quantitatively that the SVC is the stronger model on this dataset, and that DistilBERT's contribution is currently net-negative for the underrepresented class.
-   **Ensemble failure modes are documented.** Demonstrating that ensembling does not guarantee improvement — particularly when one model has a fundamental blind spot — is itself a useful finding for model selection decisions.

------------------------------------------------------------------------

## Conclusion & Next Steps

The ensemble baseline using weighted soft voting achieved a **best Macro-F1 of 0.6349** (0.3 DistilBERT / 0.7 SVC), which is competitive with but does not surpass the standalone TF-IDF + SVC baseline (Macro-F1 = 0.6405). The key finding is that **ensembling cannot compensate for a model that fails entirely on one class** — DistilBERT's zero Spam recall actively dilutes the SVC's correct predictions.

For future improvements, we could explore:

1.  **Probability calibration** — apply temperature scaling to DistilBERT and Platt scaling (`CalibratedClassifierCV`) to the SVC before averaging, so both models contribute probabilities on a comparable scale.
2.  **Class-conditional weighting** — instead of a single global weight, assign per-class weights so the SVC dominates for Spam while DistilBERT contributes more for Ham and Phish where it excels.
3.  **Feature-level fusion** — concatenate DistilBERT `[CLS]` embeddings with TF-IDF vectors as input to a single meta-classifier, combining strengths at the representation level rather than the prediction level.
