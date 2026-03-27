# Annotation Tutorial (CSV Workflow) — Sprint 3
This tutorial explains how to annotate our corpus using **CSV files** (easier for undergrads), and the rules (“regulations”) you must follow so we can convert your CSV back into our final **JSONL** annotations reliably.

**Author:** Marco Wang

**Date:** 2026-03-02  

**Course:** COLX 523

**Disclaimer:** This script was generated with the assistance of ChatGPT 5.2

---

## What you will annotate
For each message (`text_masked`), you will fill in:

1) **`scenario`** *(required, exactly one)*  
2) **`tactic_primary`** *(required for Phish/Spam, optional for Ham)*  
3) **`entities`** *(optional span annotations, using character offsets in `text_masked`)*  
4) **`notes`** *(optional: ambiguity/edge cases)*

> **Important:** Do **not** edit pre-filled columns like `id`, `label`, or `text_masked`.
> **Workflow note:** Annotators will complete annotations in **CSV** for ease of editing.  
> The project team will convert the returned CSV files into the final **JSONL** format described in the Annotation Schema section.

---

## CSV files you will receive
You will be given a CSV (one row per message) with at least these columns:

| Column | Who fills | Description |
|---|---|---|
| `label` | pre-filled | 0=Ham, 1=Phish, 2=Spam (do not change) |
| `text_masked` | pre-filled | text to annotate (do not change) |
| `scenario` | you | required, choose exactly one from list below |
| `tactic_primary` | you | required if label=1 or 2; optional if label=0 |
| `entities` | you | optional; list of entity spans in a strict format |
| `notes` | you | optional notes |

---

## Step-by-step workflow
1. Open the CSV in **Google Sheets / Excel** (recommended) or a text editor.
2. For each row:
   - Read `text_masked`
   - Choose **one** `scenario`
   - Choose **one** `tactic_primary` if the label is **Phish (1)** or **Spam (2)**
   - Optionally add entities in the `entities` column (format below)
   - Add `notes` if anything is ambiguous
3. Save as **CSV** (not XLSX).
4. Name your file:  
   `annotator_<yourname>_<YYYY-MM-DD>.csv`
5. Send/upload the CSV back to the team.

---

# Regulations (must follow)

## R0 — Use only allowed label strings
Only use the exact label strings listed in this tutorial for:
- `scenario`
- `tactic_primary`
- entity `TYPE` and `SUBTYPE`

Do not invent new labels. If none apply, use `OTHER` for scenario and leave `entities` blank.

## R1 — Scenario is required (exactly one)
Every row must have exactly one `scenario`.  
If unsure, use `OTHER` (last resort).

**Allowed `scenario` values**
- `HR_RECRUITING`
- `FINANCE_TRADING`
- `EXPENSE_ADMIN`
- `IT_SYSTEM`
- `LEGAL_REGULATORY`
- `SOCIAL_PERSONAL`
- `FRAUD_SOCIAL_ENGINEERING`
- `OTHER`

**Decision rule:** pick the **main purpose** of the message. If multiple topics appear, choose the dominant intent.

---

## R2 — `tactic_primary` is required for Phish/Spam
**Allowed `tactic_primary` values**
`{AUTHORITY, URGENCY, THREAT, REWARD, HELPFUL_SERVICE}`

**Requirement rule**
- If `label` is `1` (Phish) or `2` (Spam): `tactic_primary` **must not be blank**
- If `label` is `0` (Ham): `tactic_primary` may be blank (treated as `null`) unless the tactic is very clear

**Decision rule:** choose the **dominant** tactic (no multi-labeling).

---

## R3 — Entities must follow the exact format
Entities are annotated on **`text_masked`** using **character offsets**.

### Entity types
**Core types (most common)**
- `PERSON`
- `ORG`
- `ROLE_TITLE`
- `DATE`
- `TIME`
- `LOCATION`
- `ATTACHMENT`
- `MONEY`
- `ID`
- `SYSTEM_TOKEN`

**Optional types (only if explicitly present)**
- `URL`
- `EMAIL`
- `PHONE`
- `CREDENTIAL`

### Optional subtypes (use only when obvious)
Use these subtypes only when clearly applicable:
- HR: `RESUME_DOC`, `INTERVIEW_EVENT`
- Expense: `EXPENSE_REPORT_ID`, `AMOUNT_DUE`, `APPROVAL_LINK`
- IT: `FILE_PATH`, `ERROR_MESSAGE`, `SYSTEM_NAME`
- Legal: `REGULATOR`
- Social: `VENUE`

---

## R4 — Entity encoding in CSV (strict)
Put all entities for a row into the `entities` column using this format:

### Format
Each entity =  
`start:end:TYPE`  
or  
`start:end:TYPE:SUBTYPE`

Multiple entities are separated by **semicolon** `;`

### Example (with subtype)
`11:22:PERSON;120:129:ATTACHMENT:RESUME_DOC;340:345:SYSTEM_TOKEN`

### Example (URL subtype)
`55:60:URL:APPROVAL_LINK`

**Rules**
- Offsets use Python slicing convention: `text_masked[start:end]`
- No spaces inside the entity string
- Use `;` to separate entities (do not use commas)
- If you include a subtype, it must be one of the allowed subtypes above
- If you cannot confidently set offsets, leave `entities` blank and add a note

---

## R5 — Do not change the text
- Do not edit `text_masked`
- Do not add or remove characters
- Do not “fix typos”
- If you spot problems (e.g., weird placeholders like `<NUM>`), note them in `notes`

---

## R6 — CSV formatting rules (to avoid conversion failures)
- Keep the file as **CSV** (not XLSX)
- Do not add new columns
- Do not rename columns
- Avoid line breaks inside cells
- If your spreadsheet auto-adds quotes, that’s fine—just keep structure consistent

---

# Offsets: how to get `start:end`
Offsets are character positions in `text_masked`.

**Quick method (recommended):**
- Copy `text_masked` into a plain text editor (e.g., VS Code)
- Find the exact substring you want to label
- Use editor tools or a team-provided helper to get character index positions  
  *(If you cannot reliably get offsets, leave `entities` blank and write a note.)*

---

# Examples

## Example 1 (Ham — HR Recruiting)
text_masked: `Please schedule an interview with Alex on Friday. Resume attached: resume.doc`

scenario: `HR_RECRUITING`  
tactic_primary: *(blank)*  
entities: `33:38:DATE:INTERVIEW_EVENT;63:73:ATTACHMENT:RESUME_DOC`

## Example 2 (Phish)
text_masked: `Your bank account is locked. Verify immediately: <URL>`

scenario: `FRAUD_SOCIAL_ENGINEERING`  
tactic_primary: `URGENCY`  
entities: `5:9:ORG;25:31:THREAT;33:41:SYSTEM_TOKEN;53:58:URL`

---

# Before you submit (checklist)
- [ ] Every row has `scenario`
- [ ] All rows with `label` = 1 or 2 have `tactic_primary`
- [ ] `entities` follows `start:end:TYPE(:SUBTYPE)` with `;` separators
- [ ] Offsets match the exact substring in `text_masked`
- [ ] Saved as CSV and named `annotator_<name>_<YYYY-MM-DD>.csv`

---

## Contact
If you encounter uncertain cases, leave `entities` blank and write your reasoning in `notes` so we can adjudicate consistently.