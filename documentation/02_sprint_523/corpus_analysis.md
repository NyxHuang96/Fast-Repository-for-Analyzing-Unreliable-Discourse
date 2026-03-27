---
title: "corpus_analysis.md"
author: "Yusen Huang"
date: "2026-02-28"
output: pdf_document
---

**Source Dataset:** <https://www.kaggle.com/datasets/akshatsharma2/the-biggest-spam-ham-phish-email-dataset-300000>

**Processed Corpus:** <https://drive.google.com/drive/folders/1wv_PNSCYvKHw3lGeOHqk_go-D1LiEKft?usp=sharing>

**Data Format:** JSON

**Metadata:**

| Field | Type | Description |
|------------|------------|-------------------------------------------------|
| Label | String | The **classification metadata**. It categorizes the text into specific classes: `Ham` (legitimate), `Spam`, or `Phish`. |
| Text | String | The **raw text**. This is the actual email or message content. |

## **1. Overview**

This report provides an in-depth statistical analysis of the processed email corpus, which contains 72,609,542 **tokens**. The analysis compares the linguistic properties of **Ham**, **Spam**, and **Phish** categories to identify structural and lexical differences.

## **2. Lexical Diversity (Type-Token Ratio)**

The Type-Token Ratio (TTR) indicates the vocabulary variety within each category.

|              |                            |
|--------------|----------------------------|
| **Category** | **Type-Token Ratio (TTR)** |
| **Ham**      | 0.006962                   |
| **Phish**    | **0.068335**               |
| **Spam**     | 0.016507                   |

-   **Observation:** The **Phish** category demonstrates the highest lexical diversity, with a TTR nearly **10 times higher** than Ham. This suggests that phishing emails in this dataset utilize a broader, more varied vocabulary—likely due to the inclusion of unique URLs, technical identifiers, and obfuscated strings designed to bypass filters.

-   **Observation:** **Ham** has the lowest TTR, indicating high levels of redundancy. In a large corpus, this is often caused by repetitive structural elements such as corporate signatures, legal disclaimers, and standardized headers.

## **3. Average Text Length**

Text length often correlates with the intent of the sender.

|              |                          |
|--------------|--------------------------|
| **Category** | **Avg. Length (Tokens)** |
| **Ham**      | **344.23**               |
| **Spam**     | 211.82                   |
| **Phish**    | 121.35                   |

-   **Analysis:** **Ham** messages are the longest on average, reflecting the explanatory and contextual nature of legitimate professional correspondence.

-   **Analysis:** **Phish** messages are the shortest (avg. 121 tokens). This aligns with the "urgency" tactic often used in phishing, where the goal is to provide just enough information to provoke a quick, unthinking click on a malicious link.

## **4. N-gram and Structural Analysis**

The top N-grams reveal the primary linguistic "skeletons" of each category.

### **Ham: Professional & Procedural**

-   **Top Trigram:** `('original', 'message', 'from')`

-   **Significance:** This highlights the heavy presence of forwarded email chains and professional threads within the legitimate dataset.

-   **Noise:** The frequent appearance of `escapenumber` and `escapelong` suggests that Ham contains many dates, times, or numerical figures, typical of business reporting (e.g., "reclassify merchant assets").

### **Spam: Promotional & Noisy**

-   **Top Trigram:** `('\x81', '\x81', '\x81')` and `('_', '_', '_')`

-   **Significance:** The presence of hex codes and repeated underscores indicates "noise" or encoding artifacts common in mass-marketing emails that attempt to bypass Bayesian filters through character obfuscation.

### **Phish: Temporal & Newsletter-like**

-   **Top Bigrams:** `('aug', '2008')`, `('top', '10')`, `('cable', 'news')`

-   **Significance:** Unlike the "account verification" Phish samples seen previously, this specific data subset appears to mimic **newsletters** or **daily updates** (e.g., "Daily Top 10", "Cable News Network"). This indicates a sophisticated spoofing strategy where malicious content is embedded within seemingly harmless periodic updates.

## **5. Conclusion**

The statistical signatures of these categories are distinct:

1.  **Ham** is characterized by long, repetitive, and procedural language.

2.  **Spam** is characterized by promotional noise and encoding artifacts.

3.  **Phish** is characterized by short, diverse, and highly specific spoofs (in this case, mimicking news media) with high lexical variety.
