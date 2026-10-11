# 🕵️‍♂️ The *-Files: Anonymisation Pipeline

**Team Submission for NOS - JunctionX Lisbon 2026**
**(Nova Finance Club — João Henriques, Rogério Soares, and Salvador Antunes)**

This repository contains our complete, end-to-end solution for the NOS
Generative AI Hackathon. The challenge is to anonymise
`raw_data/document_to_anonymize.pdf`, a Portuguese medical admission
report with fictitious personal data. All solution code, prompts, and
scored artefacts live in `submission/`. Every command below is run from
the **repository root**, because modules use absolute `submission.*`
imports.

## 💡 What it Does

Our pipeline takes the challenge PDF and redacts sensitive information
across 5 categories defined in `submission/anonymizer/categories.json`:
Identity, Contact/Location, Official/Financial IDs, Health, and
Sensitive Social Data. Health data is masked by team decision.

**Our Core Philosophy: "LLMs find, Python masks."**
To guarantee zero hallucinated changes to layout, spacing, or accents,
we decoupled detection from redaction:
1. **Gemini** is used strictly as a reasoning engine to identify *what*
   to mask and on what line (`Finding`: text, category, reason, context).
2. **Plain Python** (`anonymizer/masking.py`) executes the replacement,
   swapping each sensitive word for a single asterisk (`*`) while
   preserving surrounding whitespace, punctuation, and line breaks.

Scored artefacts (do not rename, move, or delete):
- `submission/submission.txt` — only the masked document.
- `submission/prompt.txt` — the exact prompt that produced that output
  (also the shared base of every agent prompt). It ends with
  `### DOCUMENTO ###` followed by the document, so CI can run it
  single-shot.
- Solution code in `submission/` (also evaluated).

## ✨ Core Features (The 4 Pillars)

1. **Detect (Context-Aware):** A multi-agent Gemini architecture
   (Detector, Context Checker, and Reviewer) reasons about the text. It
   does not only look for formats like emails; it understands context to
   drop false alarms and catch hidden identifiers. At most **3 Gemini
   calls** per document, made one after another and spaced 13 s apart
   (free-tier keys allow as few as 5 requests per minute).
2. **Mask:** Deterministic masking replaces each sensitive word with one
   `*` (e.g. `Ana Correia` becomes `* *`, and `Flores,` becomes `*,`).
   Leading/trailing `,.;:()!?` are kept; inner `@ . - / +` belong to the
   word (`maria.santos@x.pt` → `*`). A finding with a `context` line is
   only masked inside that line (the `Lisboa` in the report title stays
   visible).
3. **Preserve:** Using PyMuPDF, the original PDF layout, line breaks, and
   accents are mapped to plain text. Empty lines and trailing spaces are
   removed as requested. The header date and reference code
   (`ADM-2025-04-15-089`) are *not* masked, while the birth date is;
   context decides what counts as sensitive.
4. **Validate:** Each run writes `submission/outputs/report.md` (readable
   table of findings, rejections, unmatched spans, and an optional
   0–100 meaning-preservation score) plus `findings.json`. Synthetic
   documents in `submission/synth/data/` are scored word-by-word
   (precision / recall / F1) against the same masker.

## 🏗️ Architecture

The orchestrator is **plain Python, not an LLM**. Agents only find
sensitive text; `apply_findings` does all masking. LLMs never rewrite
the document, since any rewording is a diff error.

Flow (at most 3 API calls per document):

* **Extraction:** `submission/anonymizer/extract.py` reads
  `raw_data/document_to_anonymize.pdf`.
* **Detector Agent:** Gemini scans the text and categorises findings.
* **Context Checker Agent:** Gemini evaluates surrounding text to add
  indirect identifiers or reject false positives (skipped if nothing
  was found).
* **First Masking Pass:** `submission/anonymizer/masking.py`.
* **Reviewer Agent:** Gemini reads the *already-masked* text to spot
  leaks.
* **Final Masking & Output:** A second Python pass writes
  `submission/submission.txt` and `submission/outputs/report.md`.

`--single-shot` sends `prompt.txt` alone, like the hidden CI prompt
check (`prompt_eval.yml`). CI never runs our code; it diffs
`submission.txt` (`evaluation.yml`) and lints with `flake8 .`
(`lint.yml`). Lines in this repo are kept at **79 characters or fewer**.

*(For detailed design decisions, see [`docs/design.md`](docs/design.md).)*

### Repository layout

```text
nos-gen-ai-hackathon/          # run every command from here
  requirements.txt
  .env                         # gitignored; see Setup
  raw_data/
    document_to_anonymize.pdf  # challenge input
  tutorials/                   # organisers' brief and helpers
  .github/workflows/           # evaluation, prompt_eval, lint
  submission/                  # all solution code and artefacts
    prompt.txt                 # hand-written; CI single-shot input
    submission.txt             # pipeline output (masked document)
    run.py                     # CLI: python -m submission.run
    validation.py              # DocumentValidator (meaning score)
    data_generator.py          # synthetic labelled documents
    anonymizer/
      config.py                # paths, API_KEY, per-role models
      llm.py                   # GeminiClient + reply cache
      schemas.py               # Finding and agent result models
      categories.json          # taxonomy (single source of truth)
      categories.py
      extract.py               # PDF -> text
      masking.py               # findings -> masked text
      base_prompt.py           # prompt.txt minus the document
      orchestrator.py          # run_pipeline()
      report.py                # findings.json + report.md
      agents/                  # detector, context_checker, reviewer
      prompts/                 # per-agent focus templates
    synth/data/                # generated *.txt / *_masked.txt pairs
    evaluate/score.py          # precision / recall / F1
    optimize/search.py         # offline prompt search (stub)
    tests/                     # unittest + FakeLLMClient (no network)
    outputs/                   # report.md, findings.json, .cache/
    docs/                      # design.md, implementation-plan.md
```

## ⚙️ Setup Instructions

### 1. Prerequisites
* Python 3.10+
* A Google Gemini API Key ([Get one here](https://aistudio.google.com/))

### 2. Installation
Clone the team's fork and stay at the repository root:
```bash
git clone https://github.com/<your-account>/nos-gen-ai-hackathon.git
cd nos-gen-ai-hackathon
```

Create and activate a virtual environment:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

Install dependencies from the **root** `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file at the **repository root** (gitignored).
`anonymizer/config.py` loads it on every lookup. The pipeline requires
`API_KEY`; model names are optional.

```env
API_KEY=your_api_key_here
MAIN_MODEL=gemini-3.8-flash
DATA_GEN_MODEL=gemini-3.8-flash
VALIDATION_MODEL=gemini-3.8-flash
```

| Variable | Role | When unset |
|---|---|---|
| `API_KEY` | Gemini authentication | error (required) |
| `MAIN_MODEL` | pipeline agents, single-shot, scoring | `gemini-3.8-flash` |
| `DATA_GEN_MODEL` | synthetic data generator | `MAIN_MODEL` |
| `VALIDATION_MODEL` | meaning-preservation judge | score skipped |

The data generator also accepts `GEMINI_API_KEY` as an alias for
`API_KEY`. Free-tier limits are tight and differ per model (seen: 20
requests/day for `gemini-3.8-flash`, 5 requests/minute for
`gemini-3.6-flash`); quotas count per Google Cloud project and model, so
a second key in the same project adds nothing. Retired models may still
appear in `models.list()` but return 404 for new keys.

## 🚀 Usage

Run these commands from the **repository root**, with `.venv` active.
Always use `python -m`; never run the files by path.

**1. Generate the Final Submission:**
Runs the full pipeline, reads `raw_data/document_to_anonymize.pdf`, and
writes the masked text to `submission/submission.txt` plus
`submission/outputs/report.md` and `findings.json`.
```bash
python -m submission.run
python -m submission.run --single-shot   # prompt.txt alone, like CI
python -m submission.run --no-validate   # skip the entailment score
python -m submission.run --no-cache      # ask Gemini again (see below)
```

Every successful Gemini reply is stored in `submission/outputs/.cache/`
(gitignored), keyed by model and prompt. A run that fails halfway (e.g.
a 503 "high demand") costs only the missing calls when rerun, and a
changed prompt is always sent fresh. `--no-cache` ignores the stored
replies and replaces them; it works for `evaluate.score` too.

**2. Evaluate against Synthetic Data:**
First generate labelled pairs (`*.txt` + `*_masked.txt`) into
`submission/synth/data/` (same categories and the same `apply_findings`
masker as the live pipeline). Then score the pipeline word by word
(precision, recall, F1).
```bash
python -m submission.data_generator --count 5
python -m submission.evaluate.score
python -m submission.evaluate.score --data path/to/pairs
```

**3. Run Test Suite:**
Runs the unittest suite with `FakeLLMClient` and a fake clock. Tests
never hit the network, sleep, or load a real `.env`.
```bash
python -m unittest discover -s submission -t .
```

Lint (CI runs `flake8 .` on every push/PR; exclude the venv locally):
```bash
pip install flake8
flake8 . --exclude .venv
```

## 🌐 Optional Validation (Meaning Preservation)

The brief allows optional manual validation of false positives and
false negatives. There is no separate web API: review happens through
the pipeline report.

1. Run `python -m submission.run` from the repo root. If
   `VALIDATION_MODEL` is set, `DocumentValidator` scores meaning
   preservation (0–100) with one extra Gemini call. `--no-validate`
   skips that call.
2. Open `submission/outputs/report.md`: masked findings (text, category,
   reason), spans the context checker rejected, and findings not found
   verbatim in the document.
3. `submission/outputs/findings.json` is the same data in machine-readable
   form. `submission/synth/` and `evaluate/score.py` measure detection
   quality without touching the hidden challenge target.

## 📚 Open-Source Libraries Used

* **google-genai:** Gemini client (`GeminiClient` in `anonymizer/llm.py`).
  Spaces requests to each model 13 s apart; a per-minute 429 waits the
  exact delay Google states, server errors (500/503, dropped
  connections) back off 5 → 60 s over 5 retries, and a per-day 429
  fails fast.
* **PyMuPDF (fitz):** High-fidelity text extraction so PDF layout maps
  to plain text.
* **pydantic:** Structured JSON output from agents (`Finding` and related
  schemas).
* **python-dotenv:** Loads the root `.env` without committing secrets.
* **unittest:** Deterministic masking, agent parsing, and pipeline tests
  with `tests/fakes.py` (`FakeLLMClient`).
* **flake8:** PEP 8 / 79-character line length, matching CI `lint.yml`.

---
*Built for JunctionX Lisbon 2026 by João Henriques, Rogério Soares, and
Salvador Antunes.*
