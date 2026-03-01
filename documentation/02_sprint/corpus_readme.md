---
title: "corpus_readme.md"
author: "Yusen Huang"
date: "2026-02-26"
output: pdf_document
---

**Source Dataset:** <https://www.kaggle.com/datasets/akshatsharma2/the-biggest-spam-ham-phish-email-dataset-300000>

**Processed Corpus:** <https://drive.google.com/drive/folders/1wv_PNSCYvKHw3lGeOHqk_go-D1LiEKft?usp=sharing>

**Data Format:** JSON

**Metadata:**

| Field | Type | Description |
|-------|--------|----------------------------------------------------------|
| Label | String | The **classification metadata**. It categorizes the text into specific classes: `Ham` (legitimate), `Spam`, or `Phish`. |
| Text | String | The **raw text**. This is the actual email or message content. |

**Total Number of Documents:** 1

**Total Amount of Text (in tokens):** 72,609,542

**Any Known Problem:** The kaggle dataset that we have downloaded online is very new and may have some problems. Although currently not known, we anticipate there may be issues with data as we do not know where it has been scraped from or how long ago. This does not have an immediate impact on our project, but this is something to consider in the future. 
