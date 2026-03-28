---
title: "split_reason.md"
author: "Yusen Huang"
date: "2026-03-28"
editor_options: 
  markdown: 
    wrap: sentence
---

## Overview

This document explained the logic behind our split reasons.

## Q&A

Q: Do you have the same speaker appear in multiple sets?

A: No, We have a `check_leakage` function to prevent same speaker appear in multiple sets.

Q: Do you have unrealistic class distributions in one or more set?

A: No, we have designed strict class balance splitting of annotation, to make sure in each set the annotation is more or less evenly distributed.

Q: Is one annotator disproportionately represented in a set?

A: This is indeed answers we do not know, because in our original annotation we do not have annotators' information stored in the data.
So it it possible that some of the annotators' data could be more in a dataset.

Q: If your data is multilingual, are the languages fairly represented?

A: We currently only have English and Chinese.
Chinese make up almost half of the total annotations.

Q: Does that fragment your data even more?

A: Yes.
The split script stratifies by label (Ham/Spam/Phish) but not by language.
So after the 80-10-10 split, there's no guarantee that English and Chinese are evenly distributed across train/dev/test.

Q: Is your split replicable (and documented)?

A: Yes, we used a fixed random seed (`581`) to ensure reproducibility across runs.

Q: If your hard-drive started sending out purple smoke, could you re-create the dataset?

A: Yes, our original annotations and split script have all been uploaded to Github Repo, even if our hard-drive is destroyed due to unfortunate accident, we can still recreate the dataset.
