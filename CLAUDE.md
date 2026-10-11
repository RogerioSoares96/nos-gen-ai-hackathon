# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working
with code in this repository.

## What this repo is

Team fork (Nova Finance Club) of `nosportugal/nos-gen-ai-hackathon` for
JunctionX Lisbon 2026 (10–11 Oct 2026). Challenge: build an LLM-based system
(suggested: Google Gemini API) that anonymizes
`raw_data/document_to_anonymize.pdf`, a Portuguese document with fictitious
personal data. Full brief: `NOS - JunctionX Lisbon 2026.pdf` (untracked) and
`tutorials/problem_and_eval.md` (Portuguese).

## Output contract (scored automatically against a hidden reference)

- Each sensitive **word** becomes a single `*`: `Ana Correia` -> `* *`.
- Keep the original formatting and meaning; the only layout change is that
  **empty lines are removed**.
- Expected start of a correct output:

  ```text
  Relatório de Admissão - Centro Médico Lisboa
  Data: 15 de abril de 2025
  Referência: ADM-2025-04-15-089
  Informações do Paciente:
  ```

  Dates and reference codes at the header are *not* masked; context decides
  what counts as sensitive.

## Submission files (do not rename, move or delete)

- `submission/submission.txt`: only the raw model output. No team name,
  explanation or extra formatting.
- `submission/prompt.txt`: the exact prompt that produced that output.
- Solution code (`.py` / `.ipynb`) goes inside `submission/`. It is also
  evaluated.

## CI (runs on PRs to upstream `main`)

- `evaluation.yml`: diffs `submission/submission.txt` against a secret target
  (script is hidden).
- `prompt_eval.yml`: scores `submission/prompt.txt` with a hidden script using
  `google-genai`.
- `lint.yml`: `flake8 .` on every push/PR with default settings. Keep every
  line in this repo at **79 characters or fewer**. Do not add a flake8 config
  to relax it. Run locally with `pip install flake8 && flake8 .` (or
  `flake8 path/to/file.py` for a single file).
- Both evaluation workflows **skip and still pass** if their file is missing,
  so a green check doesn't guarantee a score.
- `prompt_eval.yml` most likely sends `prompt.txt` to Gemini **alone, in one
  call** (inferred; the script is secret). CI never runs our code.

## Architecture (`submission/`)

Spec: `submission/docs/design.md`. Plan and status:
`submission/docs/implementation-plan.md`.

- **Flow:** `prompt.txt` (hand-written, also the shared base of every agent
  prompt) + PDF -> `extract.py` -> detector (all categories) -> context
  checker (all findings; skipped if none) -> `masking.py` -> reviewer ->
  `masking.py` -> `submission.txt` + `outputs/report.md`. At most 3
  Gemini calls per document, made one after another (free tier allows as
  few as 5 requests per minute).
- **The orchestrator is plain Python, not an LLM.** Agents only *find*
  sensitive text (`Finding`: text, category, reason, context line); the
  deterministic `apply_findings` does all masking. LLMs never rewrite the
  document, since any rewording is a diff error.
- **Masking rule:** one `*` per whitespace-separated word; leading and
  trailing `,.;:()!?` kept; inner `@ . - / +` belong to the word
  (`maria.santos@x.pt` -> `*`); whole-word matches only; a finding with a
  `context` line is only masked inside that line (keeps the `Lisboa` of the
  report title visible).
- **`anonymizer/categories.json`** is the single source of truth for what is
  sensitive (5 categories; health data *is* masked, by team decision). The
  detectors, `prompt.txt` (a test checks the ids are in sync) and the
  synthetic data generator all follow it.
- **`prompt.txt`** ends with a line `### DOCUMENTO ###` followed by the
  document; `base_prompt.load_base` drops the document for the agents.
- **Synthetic data** (`data_generator.py`, Salvador): writes
  `synth/data/*.txt` + `*_masked.txt`, masked with the same
  `apply_findings`. `evaluate/score.py` compares our output word by word
  (precision / recall / F1). `optimize/search.py` is still a stub.
- **`validation.py`** (Rogério): LLM judge for meaning preservation (0-100).

## Commands (repo root, `.venv` active)

All modules use absolute `submission.*` imports, so always run them with
`python -m`, never by file path.

- Tests: `python -m unittest discover -s submission -t .`
- Lint: `flake8 . --exclude .venv` (a bare `flake8 .` also lints `.venv`)
- Pipeline: `python -m submission.run` (`--single-shot` sends `prompt.txt`
  alone, like CI; `--no-validate` skips the judge)
- Synthetic data: `python -m submission.data_generator --count N`
- Scoring: `python -m submission.evaluate.score`

## Configuration and Gemini

- `.env` (gitignored, never read it): `API_KEY`, plus per-role models
  `MAIN_MODEL`, `DATA_GEN_MODEL` (falls back to `MAIN_MODEL`) and
  `VALIDATION_MODEL` (optional; judge skipped if unset). Read them only
  through `anonymizer/config.py`, which loads `.env` on every lookup;
  `DEFAULT_MODEL` is the single hardcoded fallback.
- `anonymizer/llm.py` (`GeminiClient`, `google-genai`) is the one shared
  client. Retries 429/500/503 and dropped connections; fails fast on a
  per-day quota 429.
- **Free tier limits are tight** and differ per model (seen: 20 requests/day
  for `gemini-3.8-flash`, 5 requests/minute for `gemini-3.6-flash`).
  Retired models (e.g. `gemini-2.5-flash`) still appear in
  `models.list()` but return 404 for new keys.

## Conventions

- Lines <= 79 characters everywhere (JSON descriptions in
  `categories.json` are the agreed exception).
- Write files as UTF-8 with `\n` line endings (`report.write_utf8`); on
  Windows, text mode would write `\r\n` and break the diff.
- Tests are `unittest`, with `tests/fakes.FakeLLMClient` (routes on prompt
  content, never on call order). Tests must never load the real `.env`:
  patch `submission.anonymizer.config.load_dotenv`.
- Ownership: core pipeline (João), `validation.py` (Rogério),
  `data_generator.py` (Salvador). Tell the owner when you change their file.

## Collaboration constraints

Collaboration (30%) and GitHub quality (5%) are judged from the fork's
history: every member commits from their own account, work happens on
branches (never directly on `main`), and history must be preserved (no
squashing or force-pushing).
