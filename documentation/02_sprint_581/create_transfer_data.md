---
title: "Transfer Learning - Option B: Secondary Data/Adjacent Tasks"
author: Darwin Zhang
date: "2026-04-04"
---

### Technical Implementation: The Baseline Pipeline
The traditional machine learning pipeline was designed to handle bilingual data without a language barrier. We utilized a Linear Support Vector Machine (SVM) combined with TF-IDF vectorization (in Sprint 1) to convert text into numerical features, capturing both single words and two-word phrases (bigrams). To support our Chinese data subset, we integrated specialized Jieba tokenizers into a unified, Whoosh-powered backend.

### Baseline Training Code
```Python
# Establishing the training pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        tokenizer=bilingual_tokenizer, 
        ngram_range=(1, 2)
    )),
    ('clf', LinearSVC(class_weight='balanced'))
])
```

### Training on the Gold training set
```Python
pipeline.fit(X_train, y_train)
```

### Performance Results and Matrix Analysis
The model was evaluated on a strictly isolated Gold Validation set to ensure accuracy and prevent data leakage. The model achieved an overall accuracy of 83%.

### Classification Report
```
Ham (Legitimate): 1.00 Precision / 1.00 Recall

Phish (Malicious): 0.71 Precision / 1.00 Recall

Spam (Commercial Junk): 1.00 Precision / 0.50 Recall
```

### Confusion Matrix
```
True Label \ Predicted	Pred_Ham	Pred_Phish	Pred_Spam
True_Ham	3	0	0
True_Phish	0	5	0
True_Spam	0	2	2
```

The results indicate that while the model perfectly identifies legitimate communication, it struggles with a specific linguistic "blind spot" where it misclassifies Spam as Phishing. This "Phish Paranoia" suggests the model over-weights urgency markers common to both categories.

### Transfer Learning Experiment: Option B (Secondary Data)
### The Strategy: Source Tagging and Vocabulary Expansion

To address the model's confusion between Spam and Phish, we hypothesized that our 105-sample Gold training set lacked the vocabulary breadth to fully understand the variations of spam.

We implemented a Transfer Learning approach by introducing 999 "Silver" records sampled from our Kaggle corpus (333 per class). To prevent this noisier data from swamping our high-precision annotations, we engineered a binary "switch" feature by appending source tokens (_source_silver_ and _source_gold_) to the text. This allowed the SVM to learn general linguistic patterns from the massive Silver data while anchoring its final decisions on the Gold data.

Training Set: 1,104 samples (105 Gold + 999 Silver)

Evaluation Set: 12 Gold samples (Unchanged, tagged with _source_gold_)

### Results

The augmented model achieved the exact same result as the baseline: 83% Accuracy, with the exact same confusion matrix (2 True_Spam misclassified as Phish).

Why did the performance not improve? This result exposes the fundamental limitation of traditional Bag-of-Words and TF-IDF models: they lack semantic awareness.

By adding 999 Silver samples, we successfully expanded the model's vocabulary, but we could not give it contextual reasoning. Both aggressive Spam and malicious Phishing share high-frequency urgency markers ("click here," "account," "urgent"). Because the SVM only weighs word presence, not word relationships or pragmatic intent, it hits a blockage in terms of sematics. When faced with highly aggressive Spam, the SVM will always conservatively guess "Phish" based on those shared trigger words.

### Conclusion

This transfer learning experiment proved that our baseline was not suffering from a lack of vocabulary (data volume), but rather a lack of contextual understanding (model architecture).

This perfectly justifies our transition in the next sprint to Deep Learning and context-aware Neural Networks, which will be able to read the surrounding context to differentiate between aggressive marketing and actual deceptive engineering.