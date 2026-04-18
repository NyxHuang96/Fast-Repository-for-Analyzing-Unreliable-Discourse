---
title: "Data Augmentation: Bootstrapping and Pseudo-Labeling"
author: Darwin Zhang
date: "2026-04-18"
Disclaimer: This documentation and code for the bootstrap code and documentation was supported by ChatGPT and Google AI overview 
---

### Methodology

The documentation here will outline how our `bootstrap.py` script addressed the specific requirements of the Sprint 4 Bootstrapping rubric.

#### 1. "You should have extra data from last block that you never got around to annotating."
* **Method:** We utilized a newly provided data file (`new_annotations_YH.jsonl`). During the data-loading phase, our script only extracted the raw `"text"` field and completely dropped the `"label"`, `"scenario"`, and `"entities"` fields. 

* **Justification:** To the model, these texts were completely raw and unannotated. We utilized this specific file rather than the massive 50,000+ raw corpus because it secretly preserves the human labels for the subsequent "Active Learning" phase, allowing us to simulate human re-annotation efficiently.

#### 2. "Randomly select a dataset equivalent to 25 percent of your annotated data from last block."
* **Method:** Our original, manually annotated "Gold" dataset consists of 105 samples. 25% of 105 is approximately 26 samples.
* **Code Implementation:** We utilized Python's `random.sample(yh_pool, 26)` to extract exactly 26 random, blinded texts from the unannotated pool. (We included the option for this as if we need to use the entire dataset, we are able to select sizes, and makes the code more robust)

#### 3. "Use your best existing model to perform inference on this dataset."
* **Method:** Our highest-performing architecture to date is the Sprint 2 Traditional Baseline (TF-IDF + Linear SVM), which achieved a 91.67% accuracy. We trained this `best_model` on our existing 1,104-sample dataset.
* **Code Implementation:** We executed `best_model_model.predict(X_unseen)` on the 26 raw texts, forcing the model to generate its own tags based entirely on its established decision boundary.

#### 4. "Use the tags provided by your model, and add the data to your training set."
* **Method:** We accepted the `best_model` pseudo-labels as ground truth for this phase.
* **Code Implementation:** We looped through the 26 predictions and appended them to our existing training array. To maintain data provenance, these new records were tagged with a metadata flag: `"source": "bootstrapped_model_guess"`. The total training set size successfully expanded from 1,104 to 1,130 samples.

#### 5. "Re-train and evaluate this model on the dev set."
* **Method:** We instantiated a brand new, blank `train_model` SVM pipeline to ensure no latent data leakage from the initial model.
* **Code Implementation:** We fit the `train_model` to the newly augmented 1,130-sample dataset and evaluated it against our strictly isolated 12-sample Dev Set.

---

### Evaluation Results

The evaluation of the bootstrapped `train_model` yielded the following results:

* **Overall Accuracy:** 0.9167 (91.67%)
* **Ham F1-Score:** 1.00
* **Phish F1-Score:** 0.91
* **Spam F1-Score:** 0.86

```text
Bootstrapping Classification Report
              precision    recall  f1-score   support

         Ham       1.00      1.00      1.00         3
       Phish       0.83      1.00      0.91         5
        Spam       1.00      0.75      0.86         4

    accuracy                           0.92        12
   macro avg       0.94      0.92      0.92        12
weighted avg       0.93      0.92      0.91        12