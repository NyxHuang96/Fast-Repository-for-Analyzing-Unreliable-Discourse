---
title: "references.md"
author: "Darwin Zhang"
date: "2026-02-26"
---

## Sprint 2 Documentation: Domain Research & Source Compilation

### Gaining Domain Expertise
To effectively identify and classify social engineering tactics, it is crucial to understand how scams are actively tracked and reported. We are leveraging several key resources to build our domain expertise, studying their methodologies for spotting and categorizing malicious messages. 

* **Government Guidance:** [Spotting malicious email messages, Government of Canada](https://www.cyber.gc.ca/en/guidance/spotting-malicious-email-messages-itsap00100) — This provides a formal, structural framework for identifying indicators of compromise in email communications.
* **Tracking Forums:** [Scam Warners - Public Forum](https://scamwarners.com/scam-message-examples) — This forum offers a comprehensive look at how communities track, report, and dissect scam message templates in real time.

### Real-World Data & Community Discussions
To complement formal guidelines, we are observing social engineering as it happens in the wild. Reviewing public discussions allows us to capture dynamic, multilingual, and newly emerging phishing tactics.

* We have reviewed and extracted real-world phishing examples directly from community subreddits (`r/phishing` and `r/scams`). 
* The initial processed data from these community forums has been stored locally at: `data/raw/reddit_processed.jsonl`

### Pre-Existing Datasets & Academic References
Our repository README has been updated to centralize all foundational data sources and relevant literature.

* **Benchmarks:** We have integrated the [Phishing Websites Dataset (UCI Machine Learning Repository)](https://archive.ics.uci.edu/dataset/327/phishing+websites) to serve as a pre-existing baseline for our project.
* **Papers & Citations:** All associated academic papers for our datasets and primary research links have been compiled and added to the project references.