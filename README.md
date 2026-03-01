# Multilingual Social Engineering and Fraud Tagging Corpus
Tags: `Fraud Analysis` | `AI Safety` | `Social Engineering` | `Multilingual NLP` | `NER` | `Sentiment Analysis`

**[Domain: AI Safety & Fraud Analysis]** • **[Methods: NER & Sentiment]** • **[Data: Multilingual & Web Scraped]**

---

## Project Structure
```
Root
├── data/
│   ├── raw/
│   │   └── reddit_processed.jsonl
│   └── source/
│       └── resources.md
├── documentation/
│   ├── 01_sprint/
│   │   └── preprocess.md
│   ├── 02_sprint/
│   │   ├── corpus_readme.md
│   │   └── token_count.md
│   ├── 03_sprint/
│   ├── 04_sprint/
│   ├── 05_sprint/
│   ├── Project Charter.md
│   └── TeamContract.md
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── token_count.py
│   └── website_scrape.py
├── tests/
│   ├── __init__.py
│   └── test_scrape.py
├── .gitignore
├── requirements.txt
└── README.md
```
---
## Project Proposal

### Evolving Beyond Spam in the Era of AI Safety

Classic ML problems like English spam-or-ham classification are largely solved. However, the landscape of social engineering is rapidly evolving. Fraudsters now deploy hyper-targeted, emotionally manipulative, and multi-lingual attacks that evade standard filters. To address this gap in AI Safety, we are building a specialized corpus focused on the linguistic mechanics of modern fraud. By looking beyond English and annotating the synergy between psychological manipulation (sentiment/urgency) and targeted information extraction (NER), this corpus aims to service under-resourced languages and provide a critical training set for next-generation fraud detection systems.

### Data Source and Acquisition

Our corpus will be built through a hybrid approach to ensure sufficient volume and quality while managing project scope iteratively:

1. **Existing Datasets (Sprint 1 Baseline):** We are baselining our corpus using the comprehensive "Biggest Spam Ham Phish Email Dataset" (250,000+ samples) from Kaggle to establish a robust foundation of known threat patterns.
2. **Custom Scraper (Sprint 2 Expansion):** We will build a custom scraper to extract live, real-world examples of social engineering from online scam-reporting databases and public forums.

### Text Characteristics

* **Kind of Text & Genre:** Short-form digital communications, including SMS transcripts, emails, and direct messages. The register is highly deceptive, persuasive, and urgency-driven.
* **Language:** Multilingual, establishing a massive baseline in English (via Kaggle datasets) alongside a targeted collection of other languages for this task (may be Mandarin next, then extend and generalize to more under-represented languages), which will be acquired via our custom web scraper.
* **Authorship:** The texts are authored by scammers and threat actors, often utilizing social engineering tactics.
* **Document Length:** Generally short, currently ranging from 10 to 200 words per document, reflecting modern digital messaging formats.

### Filtering and Corpus Size

We are targeting texts that contain explicit attempts at social engineering (e.g., impersonation, fake alerts, requests for money/data). During the scraping phase, we will filter for posts categorized under specific fraud tags or containing standard scam keywords. By aggregating our scraped data with existing datasets, we are confident we can compile enough text to approach a "Brown-sized" corpus (~1 million words). If we find the scraped data falling short, our iterative approach allows us to expand our scraper to additional public forums in later sprints.

### Corpus Structure and Metadata

The corpus will maintain structural integrity where possible. 

* **Structure:** If the data source includes multi-turn interactions (e.g., victim and scammer chat logs), we will preserve the threaded conversational structure.
* **Metadata:** If possible, we would like to have each entry paried with with metadata for futher analysis, including:
  * `platform` (e.g., WhatsApp, Email, SMS)
  * `language_tag`
  * `date_reported`
  * `source_url`

### Annotation Plan

Our annotation scheme aims to capture how fraud is executed by focusing on two overlapping layers:

1. **Named Entity Recognition (NER):** Tagging specific entities targeted or impersonated by the scammer (e.g., `[ORG: Bank Name]`, `[MONEY]`, `[MALICIOUS_LINK]`, `[PERSONAL_INFO]`, or specific fraud archetypes.
2. **Sentiment & Tactic Tagging:** Classifying the psychological driver of the message (e.g., Urgency, Fear, Greed, Authority). 

This synergy will allow future models to learn that a message containing an `[ORG]` and extreme *Urgency* is highly indicative of fraud.

### Storage Format

The corpus and all associated annotations and metadata will be stored in JSON (or JSONL) format. This allows for hierarchical storage, making it easy to pair the raw text string with its metadata dictionary and token-level NER/Sentiment annotations without breaking the formatting.

### Potential Uses Beyond Primary Annotation

Primarily, this dataset will train classifiers to detect complex social engineering. Broadly, it has high utility in the field of AI Safety:

* **Guardrail Training:** It can be used to train Large Language Models to refuse requests to generate text that matches these deceptive patterns.
* **Multimodal Security Pipelines:** The structured textual patterns of deception identified in this corpus can serve as a baseline for broader systems. For example, these linguistic triggers can be paired with acoustic feature classifiers (like those calculating Equal Error Rates for phonemes) to detect multimodal social engineering attacks, such as synchronized text-and-speech deepfake scams.
* **Linguistic Research:** Researchers can study the linguistic differences in how scams are structured across different languages and cultures.

---

## References: 

[Spotting malicious email messages, Government of Canada](https://www.cyber.gc.ca/en/guidance/spotting-malicious-email-messages-itsap00100)

[Scam Warners - Public Forum](https://scamwarners.com/scam-message-examples)

[Phishing Websites](https://archive.ics.uci.edu/dataset/327/phishing+websites)

_Note: Project scope, specific targets, and annotation guidelines will be refined iteratively across the upcoming sprints. There may be edits or approaches beyond the proposal or listed text above. We will do our best to keep the README updated to the format of our project._