# Annotation Plan and Schema Documentation

**Author:** Marco Wang

**Date:** 2026-02-27  

**Course:** COLX 523

**Disclaimer:** This script was generated with the assistance of ChatGPT 5.2

## Dataset context (what we start from)
**Dataset:** The Biggest Spam Ham Phish Email Dataset (Kaggle)  
**Columns:**  
- `label` (target)  
- `text` (email / message content)

**Label mapping (given):**
- `0` → **Ham**
- `1` → **Phish**
- `2` → **Spam**

**Class counts (given):**
- Ham (0): 168,455  
- Phish (1): 42,845  
- Spam (2): 154,148  

---

# Annotation Plan 

## What is already labeled?
The dataset already provides **message-level classification** (`label`: ham/phish/spam).  
So our *new* annotation focuses on **how** social engineering works inside the text.

## Annotation unit
- **One message/email per instance** (document-level)

## Tactic Label (Document-level): `tactic_primary`
We annotate one primary social-engineering tactic per message.

- **Label set:**  
  `tactic_primary ∈ {AUTHORITY, URGENCY, THREAT, REWARD, HELPFUL_SERVICE}`

- **Requirement rule:**  
  - Required for `label ∈ {1 (Phish), 2 (Spam)}`
  - Optional for `label = 0 (Ham)` (may be `null`)

- **Cardinality:**  
  Exactly **one** tactic label per message when present (do not multi-label).

## Scenario Types (Document-level)
Annotators must assign **exactly one** scenario label per message.  
Use `OTHER` only if none of the categories fits.

1. `HR_RECRUITING` — resumes, interviews, candidate referrals, hiring coordination  
2. `FINANCE_TRADING` — deals/contracts, rates/prices, volumes, commodities, counterparties  
3. `EXPENSE_ADMIN` — expense reports, reimbursements, approvals, internal admin workflows (e.g., Concur)  
4. `IT_SYSTEM` — system logs, errors, file paths, scheduling/parsing failures, automated notifications  
5. `LEGAL_REGULATORY` — regulators (FERC/SEC), audits, compliance memos, policy documents, legal updates  
6. `SOCIAL_PERSONAL` — personal/social coordination, events, congratulations, non-work discussions  
7. `FRAUD_SOCIAL_ENGINEERING` — impersonation, credential requests, malicious links, urgent threats, payment instructions  
8. `OTHER` — does not match any category above

We will try to balance each categories, if any category is highly imbalance, we might slightly adjust the types. 

---

## Entity Types (Span-level on `text_masked`)

### Core Entity Types (always available)
These should cover most cases in the Kaggle email subset (many corporate/Enron-style emails) without forcing rare tags.

- `PERSON` — person name (sender/recipient/mentioned individuals)
- `ORG` — organization name (company, institution, agency)
- `ROLE_TITLE` — job titles / positions (e.g., “VP”, “developer”, “trader”)
- `DATE` — dates (e.g., “June 12”, “Friday”)
- `TIME` — times (e.g., “3pm”, “14:00”)
- `LOCATION` — cities/regions/places (e.g., “Houston”)
- `ATTACHMENT` — file names / attachments (e.g., “resume.doc”, “report.xls”, “attached”)
- `MONEY` — currency amounts / fees / premiums (even if partially masked as `<NUM>`)
- `ID` — identifiers (expense report IDs, deal numbers, employee IDs, reference numbers)
- `SYSTEM_TOKEN` — dataset artifacts / placeholders (e.g., `escapenumber` → normalize to `<NUM>`)

### Optional Entity Types (tag only if explicitly present)
- `URL`
- `EMAIL`
- `PHONE`
- `CREDENTIAL`

---

## Scenario-Specific Entity Subtypes (optional)
Use these only when clearly applicable. Store as `subtype` under an existing core `type`.
If unsure, omit the subtype and keep only the core `type`.

**HR_RECRUITING**
- `{type:"ATTACHMENT", subtype:"RESUME_DOC"}`
- `{type:"DATE"/"TIME"/"LOCATION", subtype:"INTERVIEW_EVENT"}` (only for explicit interview scheduling)

**EXPENSE_ADMIN**
- `{type:"ID", subtype:"EXPENSE_REPORT_ID"}`
- `{type:"MONEY", subtype:"AMOUNT_DUE"}`
- `{type:"URL", subtype:"APPROVAL_LINK"}` (only if a URL exists)

**IT_SYSTEM**
- `{type:"SYSTEM_TOKEN", subtype:"FILE_PATH"}`
- `{type:"SYSTEM_TOKEN", subtype:"ERROR_MESSAGE"}`
- `{type:"SYSTEM_TOKEN", subtype:"SYSTEM_NAME"}`

**LEGAL_REGULATORY**
- `{type:"ORG", subtype:"REGULATOR"}`

**SOCIAL_PERSONAL**
- `{type:"LOCATION", subtype:"VENUE"}`

### Bilingual annotation policy (EN + ZH)
Because Mandarin is a translation of English:
- **Default (fast):** annotate EN `text_masked` only; ZH is parallel training data
- **Optional (multilingual check):** annotate a smaller ZH subset (e.g., 200–300) to see if tactics/NER shift after translation

---

# Annotation Schema (JSONL)

We store data in **JSONL** (one record per line).  
Each Kaggle instance yields **two records** (EN + ZH), linked by `translation_of`.

### Schema rule: `tactic_primary`
- `tactic_primary` is a document-level field.
- Allowed values: `{AUTHORITY, URGENCY, THREAT, REWARD, HELPFUL_SERVICE}`.
- Required if `label` is `1` (Phish) or `2` (Spam); optional (`null`) if `label` is `0` (Ham).

## Record schema (English)
```json
{
  "id": "kaggle_00001234",
  "source": "kaggle_akshatsharma2_biggest_spam_ham_phish",
  "lang": "en",

  "label": 0,
  "label_name": "ham",
  "scenario": "HR_RECRUITING",

  "text": "…",
  "text_masked": "…",
  "masking_applied": true, #indicates sensitive patterns and dataset placeholders (e.g., escapenumber) were normalized (e.g., to <NUM>)

  "tactic_primary": null,

  "entities": [
    { "start": 11, "end": 22, "type": "PERSON", "subtype": null, "text": "grace rodriguez" },
    { "start": 120, "end": 129, "type": "ATTACHMENT", "subtype": "RESUME_DOC", "text": "resume.doc" },
    { "start": 210, "end": 218, "type": "LOCATION", "subtype": null, "text": "houston" },
    { "start": 340, "end": 345, "type": "SYSTEM_TOKEN", "subtype": null, "text": "<NUM>" }
  ],
  # start and end are character offsets in text_masked, using Python slice convention [start:end].

  "meta": { "platform": "email" },
  "notes": ""
}
```
## Record schema (Mandarin)
```json
{
  "id": "kaggle_00001234_zh",
  "source": "translated_from_en",
  "lang": "zh",

  "label": 0,
  "label_name": "ham",
  "scenario": "HR_RECRUITING",

  "translation_of": "kaggle_00001234",
  "translation_model": "TBD",
  "translation_date": "2026-02-28",

  "text": "…",
  "text_masked": "…",
  "masking_applied": true,

  "tactic_primary": null,

  "entities": [],
  "meta": { "platform": "email" },
  "notes": ""
}
```