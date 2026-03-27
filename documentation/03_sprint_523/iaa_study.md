# Interannotator Agreement (IAA) Study

## Description
Each annotator completed 50 manual annotations. For the IAA study, we extracted the overlapped subset, which contains **10 English items** and **10 Mandarin items** annotated by multiple annotators.

We report IAA **separately for English and Mandarin**, and for the overlap results please check documentation/03_sprint/iaa_study.raw

> **Note:** Although the project involved four annotators overall, the overlap tables used in this IAA analysis contain **three annotators**. Therefore, the results reported below are based on **three-way overlap**.

---

## Metrics

### 1. `scenario`
For `scenario`, we use **Krippendorff’s alpha (nominal)** because:
- it works for **more than two annotators**
- it is appropriate for **categorical / nominal labels**
- it can handle missing values if needed

### 2. `tactic_primary`
For `tactic_primary`, we also use **Krippendorff’s alpha (nominal)** because it is likewise a **document-level categorical label**.

Since `tactic_primary` is **required for Phish/Spam but optional for Ham**, rows with `None` were treated as **missing values**, not as a valid category.

### 3. `entities`
For `entities`, we use **pairwise exact-match span-level F1** rather than a document-level agreement coefficient. This is because entity annotation is:
- **span-based**
- **multi-label per document**
- sensitive to both **boundary differences** and **entity type differences**

An entity match is counted only if two annotators assign the same:
- `start`
- `end`
- `type`
- `subtype` (if applicable)

For each language, we compute pairwise Precision / Recall / F1 between annotators, then average the pairwise F1 scores to obtain the final entity agreement score.

---

## Results
The overlapped portion of our annotations was extracted to `iaa_study.raw`.

### English

#### Document-level labels

| Field | Metric | Score |
|---|---|---:|
| `scenario` | Krippendorff’s alpha (nominal) | **0.730** |
| `tactic_primary` | Krippendorff’s alpha (nominal) | **0.302** |

#### Entity annotations

| Pair | Precision | Recall | F1 |
|---|---:|---:|---:|
| annotator_1 vs annotator_2 | 0.419 | 0.257 | 0.319 |
| annotator_1 vs annotator_3 | 0.512 | 0.647 | 0.571 |
| annotator_2 vs annotator_3 | 0.229 | 0.471 | 0.308 |
| **Average** |  |  | **0.399** |

---

### Mandarin

#### Document-level labels

| Field | Metric | Score |
|---|---|---:|
| `scenario` | Krippendorff’s alpha (nominal) | **0.887** |
| `tactic_primary` | Krippendorff’s alpha (nominal) | **1.000** |

#### Entity annotations

| Pair | Precision | Recall | F1 |
|---|---:|---:|---:|
| annotator_1 vs annotator_2 | 0.682 | 0.577 | 0.625 |
| annotator_1 vs annotator_3 | 0.636 | 0.636 | 0.636 |
| annotator_2 vs annotator_3 | 0.615 | 0.727 | 0.667 |
| **Average** |  |  | **0.643** |

---

## Interpretation

### 1. `scenario`
Scenario annotation showed **strong agreement** overall.
- **Mandarin** achieved **α = 0.887**, indicating high reliability.
- **English** achieved **α = 0.730**, indicating moderate to strong agreement.

The lower English score mainly came from a small number of disagreements between `FRAUD_SOCIAL_ENGINEERING`, `OTHER`, and `FINANCE_TRADING`, suggesting that some English messages were more ambiguous in communicative purpose.

### 2. `tactic_primary`
Tactic annotation showed a large difference across languages.
- **Mandarin** achieved **perfect agreement** (**α = 1.000**) on the valid overlapped items.
- **English** achieved **α = 0.302**, indicating low agreement.

The English disagreements were not random; they were concentrated in repeated confusions between:
- `REWARD`
- `HELPFUL_SERVICE`
- `THREAT`

This suggests that the “dominant tactic” rule was still somewhat subjective, and that some tactic categories need clearer boundaries and more examples in the annotation guidelines.

### 3. `entities`
Entity annotation was the most difficult layer.
- **Mandarin** achieved an average pairwise exact-match F1 of **0.643**
- **English** achieved an average pairwise exact-match F1 of **0.399**

This difference likely reflects:
- boundary ambiguity
- dense entity content in some English emails
- inconsistent label usage in a few cases
- the strictness of exact-match span evaluation

Because exact-match F1 requires both the same span boundaries and the same label, even small boundary differences count as full mismatches.

---

## Effect of Label Normalization
Before finalizing the entity IAA calculation, we corrected annotation typos such as:
- `PHONE_NUMBER` → `PHONE`
- `DATEE` → `DATE`

After this normalization, the Mandarin entity agreement improved slightly, while the English entity score remained unchanged. This shows that even small schema inconsistencies can noticeably reduce exact-match agreement.

---

## Limitations
This IAA study is based on a relatively small overlap set:
- **10 English items**
- **10 Mandarin items**

Therefore, the reported scores should be interpreted as **preliminary estimates** rather than definitive measures of annotation reliability.

In addition, exact-match span evaluation is strict. Some disagreements likely reflect small boundary differences rather than fully different interpretations of the same text.

A small methodological note is important here: rows with `None` for `tactic_primary` were treated as **missing values**, not as a real category. This is the preferred analysis, because `tactic_primary` is optional for Ham in our schema. If `None` were incorrectly treated as a valid label, the English agreement score would appear artificially higher, since the two unanimous `None` rows would inflate agreement. Therefore, the reported English `tactic_primary` result (**α = 0.302**) should be considered the correct and more conservative value.

---

## Future Improvements
To improve agreement in future rounds, we would:
- provide more examples for difficult categories
- further clarify the difference between `REWARD`, `HELPFUL_SERVICE`, and `THREAT`
- normalize label names more strictly before annotation
- run a short calibration round before full annotation
- provide better tools for finding exact entity offsets
- simplify or further constrain subtype usage if needed