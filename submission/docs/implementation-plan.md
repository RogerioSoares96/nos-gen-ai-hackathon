# Anonymizer Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared core of the multi-agent anonymisation pipeline in
`submission/`, so the team can refine the prompts and agents and plug in
synthetic data, scoring and prompt optimisation.

**Architecture:** Plain-Python orchestrator. Gemini agents (1 detector, 1
context checker, 1 reviewer; see the addendum) return `Finding`s as JSON. A
deterministic masker turns each sensitive word into `*`. The hand-written
`prompt.txt` is the shared base of every agent prompt.

**Tech Stack:** Python ≥ 3.10, `google-genai`, `pydantic` v2, `pymupdf`
(`fitz`), `python-dotenv`, `unittest` (pytest also works), `flake8`.

**Spec:** `submission/docs/design.md`

## Global Constraints

- Every file in the repo has lines ≤ **79 characters** (`flake8 .` with
  default settings, no config). Indentation is 4 spaces in Python.
- All code stays inside `submission/`. The exception is the root
  `requirements.txt`, which only gets lines appended.
- Imports are absolute from the repo root, e.g.
  `from submission.anonymizer.masking import apply_findings`. This matches
  the teammate's `test_validation.py`. Commands run from the repo root.
- The API key comes from env var `API_KEY`. A local `.env` is loaded with
  `python-dotenv` (never read or print it). Models come from env
  `MAIN_MODEL` (default `DEFAULT_MODEL`, `gemini-3.8-flash`),
  `DATA_GEN_MODEL` (falls back to the main model) and `VALIDATION_MODEL`
  (optional), always read through `config.py`.
- All LLM calls run at temperature `0.0` with JSON output
  (`response_mime_type="application/json"`), except single-shot text mode.
- `prompt.txt` marker line: `### DOCUMENTO ###`.
- Masking: a word is a whitespace-separated token. Leading and trailing
  characters in `,.;:()!?` are kept, and the rest becomes one `*`.
- No git commands are run by the implementer. Each task ends with a
  **checkpoint**: João reviews the diff and commits it.

## Decisions this plan adds to the spec (confirm when reviewing)

1. **`Finding.context: str = ""`.** If every occurrence of a finding were
   masked, a `Lisboa` from the address would also mask the header
   `Centro Médico Lisboa`, which the expected output keeps visible. So
   agents copy the line the finding sits on into `context`, and only
   occurrences inside that context are masked. An empty `context` still
   means "all occurrences". This updates spec §6.1 and §7.
2. **`apply_findings` returns `MaskResult(masked, unmatched)`**, not a
   bare string. Findings whose text isn't found verbatim are reported
   instead of being silently dropped.
3. **Entry point is `python -m submission.run`**, not
   `python submission/run.py`, so absolute `submission.*` imports work
   without editing `sys.path`. This updates spec §12.
4. **Trailing whitespace on each extracted line is stripped.** PyMuPDF
   leaves a trailing space on wrapped lines (`dores `). The expected
   output shows none, but this is unverified against the hidden target.

## Review Focus

1. A finding that is a substring of a longer word (`Ana` inside `Anamnese`)
   must not mask that word. The match needs word boundaries (Task 4).
2. A sensitive word that also appears in non-sensitive places (`Lisboa` in
   the header) must only be masked inside its `context` (Task 4).
3. An LLM finding not present verbatim in the text (paraphrased, different
   case, split across a line break) must be listed as unmatched, not crash
   the run or mask anything (Task 4, reported in Task 10).
4. A Gemini 429/5xx error must be retried with backoff, then raise a clear
   error. `submission.txt` is never partially written (Tasks 2 and 10).
5. An empty or whitespace-only finding text must be ignored. Otherwise it
   matches everywhere (Task 4).

---

### Task 1: Dependencies, packages and schemas

**Files:**
- Modify: `requirements.txt` (append only)
- Create: `submission/__init__.py`, `submission/anonymizer/__init__.py`,
  `submission/anonymizer/agents/__init__.py`, `submission/tests/__init__.py`
  (all empty)
- Create: `submission/anonymizer/schemas.py`
- Test: `submission/tests/test_schemas.py`

**Interfaces:**
- Produces (pydantic v2 `BaseModel`s, in `schemas.py`):
  - `Finding(text: str, category: str, reason: str, context: str = "")`
  - `DetectorResult(findings: list[Finding])`
  - `ContextResult(additions: list[Finding], rejections: list[str])`
  - `ReviewResult(findings: list[Finding])`
  - `EntailmentScore(score: int = Field(ge=0, le=100))`

- [ ] **Step 1: Append dependencies.** Run `pip install google-genai pydantic
  python-dotenv`, then append three `name==version` lines to
  `requirements.txt`, using the versions `pip show` reports.
- [ ] **Step 2: Write the failing tests**

```python
class TestSchemas(unittest.TestCase):
    def test_finding_context_defaults_to_empty(self):
        f = Finding(text="Maria", category="identity", reason="name")
        self.assertEqual(f.context, "")

    def test_entailment_score_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            EntailmentScore.model_validate_json('{"score": 101}')

    def test_context_result_parses_json(self):
        r = ContextResult.model_validate_json(
            '{"additions": [], "rejections": ["Lisboa"]}')
        self.assertEqual(r.rejections, ["Lisboa"])
```

- [ ] **Step 3:** Run `python -m unittest submission.tests.test_schemas`.
  Expected: FAIL (ImportError).
- [ ] **Step 4:** Create the empty `__init__.py` files and `schemas.py` with
  the models above.
- [ ] **Step 5:** Run the same command. Expected: 3 tests OK.
- [ ] **Step 6: Checkpoint.** Suggested message:
  `build: add core dependencies, package layout and schemas`.

### Task 2: Config and the shared Gemini client

**Files:**
- Create: `submission/anonymizer/config.py`,
  `submission/anonymizer/llm.py`, `submission/tests/fakes.py`
- Test: `submission/tests/test_llm.py`

**Interfaces:**
- Consumes: the schemas from Task 1.
- Produces:
  - `config.py`: `api_key() -> str` (calls `load_dotenv()`, reads
    `API_KEY`, raises `RuntimeError("API_KEY environment variable is not
    set")` if it's missing); `main_model() -> str` (env `MAIN_MODEL`,
    default `DEFAULT_MODEL`), `data_gen_model() -> str`,
    `validation_model() -> str | None`, each loading `.env` first;
    constants `SUBMISSION_DIR`,
    `PROMPT_PATH`, `PDF_PATH` (`raw_data/document_to_anonymize.pdf`),
    `SUBMISSION_PATH`, `OUTPUTS_DIR`, `TEMPERATURE = 0.0`,
    `RETRY_CODES = {429, 500, 503}`, `MAX_RETRIES = 3`.
  - `llm.py`: `class LLMClient(Protocol)` with
    `generate_json(prompt: str, schema: type[T]) -> T` and
    `generate_text(prompt: str) -> str`.
    `class GeminiClient(model: str, temperature: float = 0.0,
    sleep=time.sleep)`.
  - `tests/fakes.py`: `FakeLLMClient(responder)`, where
    `responder(prompt, schema) -> BaseModel | str`. A `str` return is
    parsed with `schema.model_validate_json`, the same as the real client.
    `generate_text` returns `responder(prompt, None)`. It keeps
    `prompts: list[str]`.

- [ ] **Step 1: Write the failing tests.** Patch
  `submission.anonymizer.llm.genai.Client` and set env `API_KEY=test-key`.

```python
def test_generate_json_sends_schema_and_parses(self):
    # mocked models.generate_content returns obj with .text='{"score": 87}'
    result = GeminiClient("gemini-x").generate_json("p", EntailmentScore)
    self.assertEqual(result.score, 87)
    kwargs = generate_content.call_args.kwargs
    self.assertEqual(kwargs["model"], "gemini-x")
    self.assertEqual(kwargs["contents"], "p")
    self.assertEqual(kwargs["config"].temperature, 0.0)
    self.assertEqual(kwargs["config"].response_mime_type,
                     "application/json")
    self.assertIs(kwargs["config"].response_schema, EntailmentScore)

def test_retries_on_429_then_succeeds(self):
    # side_effect: [Err(code=429), Err(code=503), ok]; sleep is a Mock
    # -> result parsed, generate_content called 3 times, sleep(1), sleep(2)

def test_gives_up_after_max_retries(self):
    # always Err(code=429) -> raises RuntimeError mentioning "429"

def test_non_retryable_error_raises_immediately(self):
    # Err(code=400) -> re-raised, called once, sleep never called

def test_missing_api_key_raises(self):
    # env without API_KEY, load_dotenv patched -> RuntimeError
```

  `Err` is a local `Exception` subclass with a `code` attribute.

- [ ] **Step 2:** Run `python -m unittest submission.tests.test_llm`.
  Expected: FAIL.
- [ ] **Step 3:** Implement `GeminiClient`. Create `genai.Client(
  api_key=config.api_key())` once in `__init__`. `generate_json` calls
  `client.models.generate_content(model=..., contents=prompt,
  config=types.GenerateContentConfig(temperature=...,
  response_mime_type="application/json", response_schema=schema))`, then
  returns `schema.model_validate_json(response.text)`. `generate_text` does
  the same without the JSON fields and returns `response.text`. Retry
  wraps the call: an exception whose `getattr(e, "code", None)` is in
  `RETRY_CODES` sleeps `2 ** attempt` seconds (1, 2, 4) and tries again.
  After `MAX_RETRIES` retries it raises
  `RuntimeError(f"Gemini failed after retries: {e}")`.
- [ ] **Step 4:** Implement `FakeLLMClient`.
- [ ] **Step 5:** Run the tests. Expected: 5 OK.
- [ ] **Step 6: Checkpoint.** Suggested message:
  `feat: shared google-genai client with retries`.

### Task 3: Move Rogério's validator onto the shared client

**Files:**
- Modify: `submission/validation.py`, `submission/test_validation.py`

**Interfaces:**
- Consumes: `LLMClient`, `GeminiClient` (Task 2), `EntailmentScore`
  (Task 1), `FakeLLMClient`.
- Produces: `DocumentValidator(client: LLMClient | None = None)`.
  `client` is `None` → `GeminiClient(model=<VALIDATION_MODEL>)`, created
  lazily on the first `entailment` call. A missing env var raises
  `RuntimeError("VALIDATION_MODEL environment variable is not set")`.
  `entailment(original_document, last_document) -> int` and
  `check_removed_words(document, expected_count) -> bool` keep their
  current behaviour.

- [ ] **Step 1: Rewrite the tests** (keep their printed diagnostics and
  existing assertions):
  - `test_call_api_uses_environment_configuration` →
    `test_default_client_uses_validation_model`: patch
    `submission.validation.GeminiClient`, set env
    `VALIDATION_MODEL=models/test-model`, call `entailment`, assert
    `GeminiClient` was called with `model="models/test-model"`.
  - `test_missing_validation_model_raises`: env without the variable →
    `RuntimeError`.
  - `test_entailment_returns_score`: `FakeLLMClient` returns
    `'{"score": 87}'` → 87, and the recorded prompt contains both
    documents.
  - `test_entailment_rejects_scores_outside_range`: the fake returns
    `'{"score": 101}'` → `ValueError`.
  - The two `check_removed_words` tests stay as they are.
- [ ] **Step 2:** Run `python -m unittest submission.test_validation`.
  Expected: FAIL.
- [ ] **Step 3:** In `validation.py`, remove `call_api`, `_response_text`,
  `API_BASE_URL` and the now-unused imports (`json`, `requests`, `Any`).
  `entailment` builds the same prompt text, minus the "Return only valid
  JSON" lines, which the schema replaces. It then returns
  `client.generate_json(prompt, EntailmentScore).score`. Wrap every line
  to ≤ 79 characters.
- [ ] **Step 4:** Run the tests, then `flake8 submission/validation.py
  submission/test_validation.py`. Expected: 6 OK, no flake8 output.
- [ ] **Step 5: Checkpoint.** Suggested message:
  `refactor: run Rogério's validator on the shared Gemini client`.
  Tell Rogério about the change.

### Task 4: Deterministic masker

**Files:**
- Create: `submission/anonymizer/masking.py`
- Test: `submission/tests/test_masking.py`

**Interfaces:**
- Consumes: `Finding`.
- Produces: `@dataclass MaskResult(masked: str, unmatched:
  list[Finding])`, and `apply_findings(text: str, findings:
  list[Finding]) -> MaskResult`.

- [ ] **Step 1: Write the failing tests.** `F(text, context="")` is a test
  helper building a `Finding` with `category="x"`, `reason="r"`.

```python
def test_multi_word_name(self):
    r = apply_findings("Nome: Maria Conceição Oliveira Santos",
                       [F("Maria Conceição Oliveira Santos")])
    self.assertEqual(r.masked, "Nome: * * * *")

def test_edge_punctuation_kept(self):
    text = "Morada: Rua das Flores, 123, Sacavém"
    r = apply_findings(text, [F("Flores"), F("Sacavém")])
    self.assertEqual(r.masked, "Morada: Rua das *, 123, *")

def test_parenthesis_and_comma(self):
    r = apply_findings("Filhos: 2 (João, 15 anos", [F("João")])
    self.assertEqual(r.masked, "Filhos: 2 (*, 15 anos")

def test_inner_punctuation_is_one_word(self):
    text = "Email: maria.santos@emailpessoal.pt\nCC: 12345678-9ZX0"
    r = apply_findings(text, [F("maria.santos@emailpessoal.pt"),
                              F("12345678-9ZX0")])
    self.assertEqual(r.masked, "Email: *\nCC: *")

def test_phone_with_spaces(self):
    r = apply_findings("Telefone: +351 912 345 678",
                       [F("+351 912 345 678")])
    self.assertEqual(r.masked, "Telefone: * * * *")

def test_every_occurrence_without_context(self):
    r = apply_findings("Maria Santos\nA paciente Maria Santos,",
                       [F("Maria Santos")])
    self.assertEqual(r.masked, "* *\nA paciente * *,")

def test_context_limits_masking(self):  # Review Focus 2
    text = ("Relatório - Centro Médico Lisboa\n"
            "Morada: Rua das Flores, Sacavém, Lisboa")
    r = apply_findings(text, [F("Lisboa",
                       context="Morada: Rua das Flores, Sacavém, Lisboa")])
    self.assertEqual(r.masked, "Relatório - Centro Médico Lisboa\n"
                               "Morada: Rua das Flores, Sacavém, *")

def test_no_substring_match(self):  # Review Focus 1
    r = apply_findings("Ana fez anamnese", [F("Ana")])
    self.assertEqual(r.masked, "* fez anamnese")

def test_unmatched_reported(self):  # Review Focus 3
    f = F("Maria  Santos")
    r = apply_findings("Maria Santos", [f])
    self.assertEqual(r.masked, "Maria Santos")
    self.assertEqual(r.unmatched, [f])

def test_blank_finding_ignored(self):  # Review Focus 5
    r = apply_findings("Nome: Maria", [F("  ")])
    self.assertEqual(r.masked, "Nome: Maria")

def test_masking_masked_text_is_stable(self):
    r = apply_findings("Nome: * Santos", [F("Santos")])
    self.assertEqual(r.masked, "Nome: * *")
```

- [ ] **Step 2:** Run `python -m unittest submission.tests.test_masking`.
  Expected: FAIL.
- [ ] **Step 3:** Implement `apply_findings`:

```
1. masked_chars = set()
2. for each finding with text.strip():
     regions = [(0, len(text))] if no context, else every occurrence of
               context in text (if none: unmatched, continue)
     inside each region, find occurrences of finding.text with
       re.finditer(r"(?<!\w)" + re.escape(t) + r"(?!\w)")
     no occurrence anywhere -> unmatched; else add every char index
3. rebuild line by line; for each token m in re.finditer(r"\S+", line):
     if any char of m's absolute span is in masked_chars:
       lead = leading chars in ",.;:()!?", trail = trailing ones
       core empty -> keep token; else token = lead + "*" + trail
```

  Keep `"\n"` joins, so the line structure is unchanged.
- [ ] **Step 4:** Run the tests. Expected: 11 OK.
- [ ] **Step 5: Checkpoint.** Suggested message:
  `feat: deterministic word masker`.

### Task 5: PDF extraction

**Files:**
- Create: `submission/anonymizer/extract.py`
- Test: `submission/tests/test_extract.py`

**Interfaces:**
- Produces: `pdf_to_text(path: str | Path) -> str` and
  `clean_lines(raw: str) -> str`. The latter drops lines that are empty
  or only whitespace, `rstrip`s every line, and joins with `"\n"`.

- [ ] **Step 1: Write the failing tests**

```python
def test_clean_lines_drops_empty_and_trailing_space(self):
    raw = "Relatório\n\n  \nA paciente relatou dores \nabdominais\n"
    self.assertEqual(clean_lines(raw),
                     "Relatório\nA paciente relatou dores\nabdominais")

def test_real_pdf_header(self):
    text = pdf_to_text(config.PDF_PATH)
    self.assertEqual(text.splitlines()[:4], [
        "Relatório de Admissão - Centro Médico Lisboa",
        "Data: 15 de abril de 2025",
        "Referência: ADM-2025-04-15-089",
        "Informações do Paciente:"])
    self.assertIn("Nome: Maria Conceição Oliveira Santos", text)
```

- [ ] **Step 2:** Run `python -m unittest submission.tests.test_extract`.
  Expected: FAIL.
- [ ] **Step 3:** Implement with `fitz.open(path)`, concatenating
  `page.get_text()` for every page, then `clean_lines`. Don't use the
  tutorial's `remove_all_special_characters`.
- [ ] **Step 4:** Run the tests. Expected: 2 OK.
- [ ] **Step 5: Checkpoint.** Suggested message:
  `feat: PDF to text extraction`.

### Task 6: Category pseudo-DB

**Files:**
- Create: `submission/anonymizer/categories.json`,
  `submission/anonymizer/categories.py`
- Test: `submission/tests/test_categories.py`

**Interfaces:**
- Produces: `@dataclass(frozen=True) Category(id: str, name: str,
  description: str, examples: list[str], do_not_mask: list[str])`, and
  `load_categories(path: Path = CATEGORIES_PATH) -> list[Category]`.

- [ ] **Step 1: Write the failing tests:**
  - `test_loads_five_categories_in_order`: the ids are `["identity",
    "contact_location", "gov_financial_ids", "health",
    "sensitive_social"]`.
  - `test_ids_unique_and_fields_present`: every `description` is
    non-empty, and every entry has at least one example.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Write `categories.json` as a JSON list, with content from
  spec §8. Use examples that are not taken from the real document (e.g.
  `"Pedro Almeida"`), so the prompts don't leak the answer. The
  `do_not_mask` entries cover the report date, the reference code, the
  institution name in the title, and section headings. Write the loader.
- [ ] **Step 4:** Run the tests. Expected: 2 OK.
- [ ] **Step 5: Checkpoint.** Suggested message:
  `feat: category pseudo-database`.

### Task 7: Base prompt, agent prompt templates and the first `prompt.txt`

**Files:**
- Create: `submission/anonymizer/base_prompt.py`,
  `submission/anonymizer/prompts/detector.md`, `context_checker.md`,
  `reviewer.md`
- Modify: `submission/prompt.txt` (first draft)
- Test: `submission/tests/test_base_prompt.py`

**Interfaces:**
- Consumes: `Category`, `load_categories`, `config.PROMPT_PATH`.
- Produces:
  - `MARKER = "### DOCUMENTO ###"`
  - `load_base(path: Path = PROMPT_PATH) -> str`: returns the text before
    the marker, stripped, or the whole file if there is no marker.
  - `compose(base: str, agent: str, document: str, **fields: str) ->
    str`: `base + "\n\n" + Template(prompts/<agent>.md).substitute(
    fields) + "\n\n" + MARKER + "\n" + document`. It uses
    `string.Template` (`$category`), so JSON braces in the templates
    need no escaping.
  - Template fields: detector uses `$category_block`; context_checker
    uses `$category_block` and `$findings_block`; reviewer uses none.

- [ ] **Step 1: Write the failing tests**

```python
def test_load_base_drops_document(self):  # tmp file
    # "Instr\n### DOCUMENTO ###\nRelatório" -> "Instr"

def test_load_base_without_marker_returns_all(self):
    # "Instr only" -> "Instr only"

def test_compose_orders_parts(self):
    p = compose("BASE", "detector", "DOC", category_block="CAT")
    self.assertTrue(p.startswith("BASE"))
    self.assertIn("CAT", p)
    self.assertTrue(p.endswith("### DOCUMENTO ###\nDOC"))

def test_prompt_txt_mentions_every_category(self):  # spec sync test
    text = PROMPT_PATH.read_text(encoding="utf-8")
    for c in load_categories():
        self.assertIn(c.id, text)

def test_prompt_txt_ends_with_document(self):
    # text after MARKER starts with the expected header line
```

- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement `base_prompt.py`.
- [ ] **Step 4:** Write the templates, in Portuguese to match the document.
  Each one states the agent's single job and the JSON shape it returns.
  - **detector:** "find only data of this category". `text` and
    `context` must be copied character for character from the document.
    `context` is the full line.
  - **context_checker:** given `$findings_block`, return words that
    indirectly reveal those entities as `additions`, and texts that are
    not actually sensitive as `rejections`. Never rewrite anything.
  - **reviewer:** the document is already masked (`*`). Return any
    sensitive data still visible.
- [ ] **Step 5:** Write the `prompt.txt` draft following spec §9: role,
  output rules (one `*` per word, keep layout, drop empty lines, output
  only the document), the 5 categories with their ids, a do-not-mask list,
  one short invented example, then `### DOCUMENTO ###` and the output of
  `pdf_to_text(PDF_PATH)`.
- [ ] **Step 6:** Run the tests. Expected: 5 OK.
- [ ] **Step 7: Checkpoint.** Suggested message:
  `feat: base prompt, agent templates and first prompt.txt`.

### Task 8: The three agents

**Files:**
- Create: `submission/anonymizer/agents/detector.py`,
  `context_checker.py`, `reviewer.py`
- Test: `submission/tests/test_agents.py`

**Interfaces:**
- Consumes: `compose`, `Category`, `LLMClient`, `DetectorResult`,
  `ContextResult`, `ReviewResult`, `Finding`.
- Produces:
  - `detect(text: str, category: Category, base: str, client:
    LLMClient) -> list[Finding]`
  - `check_context(text: str, category: Category, findings:
    list[Finding], base: str, client: LLMClient) -> ContextResult`
  - `review(masked_text: str, base: str, client: LLMClient) ->
    list[Finding]`
  - `format_category(c: Category) -> str` (in `detector.py`, reused by
    `context_checker`): id, name, description, examples, do-not-mask.

- [ ] **Step 1: Write the failing tests** with `FakeLLMClient`:
  - `test_detect_forces_category_id`: the fake returns a finding with
    `category="wrong"`, and `detect` returns it with `category="health"`.
    The agent's category is authoritative.
  - `test_detect_prompt_contains_category_and_document`.
  - `test_check_context_lists_findings_in_prompt`: each finding's `text`
    appears in the recorded prompt.
  - `test_review_sends_masked_text`: the prompt ends with the masked
    text.
  - `test_review_findings_keep_their_category`.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement the agents. Each is one `compose` call plus one
  `client.generate_json` call.
- [ ] **Step 4:** Run the tests. Expected: 5 OK.
- [ ] **Step 5: Checkpoint.** Suggested message:
  `feat: detector, context checker and reviewer agents`.

### Task 9: Orchestrator

**Files:**
- Create: `submission/anonymizer/orchestrator.py`
- Test: `submission/tests/test_orchestrator.py`

**Interfaces:**
- Consumes: the agents (Task 8), `apply_findings`/`MaskResult` (Task 4),
  `load_categories`.
- Produces: `@dataclass PipelineResult(masked: str, findings:
  list[Finding], rejected: list[str], unmatched: list[Finding])`, and
  `run_pipeline(text: str, base: str, client: LLMClient, categories:
  list[Category] | None = None, max_workers: int = 5) -> PipelineResult`.

- [ ] **Step 1: Write the failing tests.** The fake responder routes on
  the prompt content (category id / agent), so the order of parallel calls
  doesn't matter:
  - `test_end_to_end_masks_detected_and_reviewed`: text `"Nome: Maria
    Santos\nReligião: Católica"`. The identity detector returns `Maria
    Santos`, the reviewer returns `Católica`. Expect `"Nome: * *\nReligião:
    *"`.
  - `test_rejections_remove_findings`: the context checker rejects
    `"Santos"`, so it stays visible.
  - `test_context_checker_only_called_for_categories_with_findings`:
    count the context prompts.
  - `test_findings_deduplicated`: two detectors return the same
    `(text, context)`, and `result.findings` holds it once.
  - `test_call_budget`: total calls ≤ 2 × number of categories + 1.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement the flow from spec §4:
  1. Run the detectors in a `ThreadPoolExecutor`.
  2. Run the context checkers for categories with findings, also in
     parallel.
  3. Merge the additions, drop findings whose text is in the rejections,
     and dedupe on `(text, context)`.
  4. Mask (v1), run the reviewer on v1, then mask v1 again with the
     reviewer's findings.
  5. Collect `unmatched` from both masking passes.
- [ ] **Step 4:** Run the tests. Expected: 5 OK.
- [ ] **Step 5: Checkpoint.** Suggested message:
  `feat: pipeline orchestrator`.

### Task 10: Report and CLI

**Files:**
- Create: `submission/anonymizer/report.py`, `submission/run.py`
- Modify: `submission/docs/design.md` (§6.1 `context` field and §7
  context rule, `MaskResult`, §12 `python -m submission.run`)
- Test: `submission/tests/test_report.py`, `submission/tests/test_run.py`

**Interfaces:**
- Consumes: everything above, plus `DocumentValidator`.
- Produces:
  - `write_report(result: PipelineResult, out_dir: Path, entailment:
    int | None = None) -> None`. It writes `findings.json` (findings,
    rejected, unmatched as lists of dicts) and `report.md` (a table with
    text | category | reason, then the unmatched list, then
    `Meaning preservation: <n>/100` or `not computed`).
  - `main(argv: list[str] | None = None, client: LLMClient | None =
    None) -> int`, in `run.py`. Flags: `--single-shot`, `--no-validate`.

- [ ] **Step 1: Write the failing tests:**
  - `test_report_files_written` (temp dir): both files exist, and
    `findings.json` round-trips.
  - `test_report_shows_unmatched_and_score`.
  - `test_main_writes_submission_only_on_success`: patch the paths into a
    temp dir and use a fake client. `submission.txt` equals
    `result.masked`.
  - `test_main_failure_leaves_submission_untouched` (Review Focus 4): the
    fake raises `RuntimeError`, `main` returns 1, and `submission.txt`
    keeps its old content.
  - `test_single_shot_writes_outputs_file`:
    `outputs/single_shot.txt` is written, and `submission.txt` is not
    touched.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement. The pipeline path is: `pdf_to_text` →
  `load_base` → `run_pipeline` → write `submission.txt` (only after
  success, UTF-8, no trailing newline added) → entailment, unless
  `--no-validate` is set or `VALIDATION_MODEL` is missing → `write_report`.
  Single-shot mode sends the **whole** `prompt.txt` through
  `generate_text`, like CI does. Exceptions are caught in `main`: print to
  stderr and return 1. Add `if __name__ == "__main__":
  sys.exit(main())`.
- [ ] **Step 4:** Update the spec sections listed under Files.
- [ ] **Step 5:** Run the tests. Expected: 5 OK.
- [ ] **Step 6: Checkpoint.** Suggested message:
  `feat: CLI entry point and explainability report`.

### Task 11: Scoring against Salvador's synthetic data

Salvador's `submission/data_generator.py` (now on `apply_findings` and
`categories.json`) writes pairs of files to `submission/synth/data/`:
`synthetic_document_NNN.txt` (the document) and
`synthetic_document_NNN_masked.txt` (the ground truth, masked under the
project's rule). Both end with one `"\n"`. Scoring compares our masked
output with the ground truth **word by word**, the same way the hidden CI
diffs `submission.txt`. Category labels aren't needed for that.

**Files:**
- Create: `submission/evaluate/__init__.py`, `submission/evaluate/score.py`,
  `submission/optimize/__init__.py`, `submission/optimize/search.py`,
  `submission/synth/README.md`
- Modify: `submission/docs/design.md` (§6.5 synthetic format, §11 inputs,
  §13 ownership)
- Test: `submission/tests/test_score.py`

**Interfaces:**
- Consumes: `run_pipeline` (Task 9), `load_base` (Task 7),
  `GeminiClient`/`LLMClient` (Task 2), `config.SUBMISSION_DIR`.
- Produces, in `evaluate/score.py`:
  - `score_masked(predicted: str, truth: str) -> dict[str, float]`, with
    keys `tp`, `fp`, `fn`, `precision`, `recall`, `f1`. A masked word is
    a whitespace token containing `*`, compared by (line, token)
    position. If the line or token counts differ, it raises
    `ValueError` naming the first mismatching line. Both maskings keep
    tokens 1:1, so a mismatch means the text itself changed. An empty
    denominator gives a precision or recall of `1.0`, and `p + r == 0`
    gives an F1 of `0.0`.
  - `load_dataset(data_dir: Path) -> list[tuple[str, str, str]]`, as
    `(stem, text, masked)` sorted by stem. It pairs `X.txt` with
    `X_masked.txt`, ignores `*_sensitive_to_mask.txt`, and strips the
    single trailing `"\n"`.
  - `evaluate(dataset, base: str, client: LLMClient) -> dict`, as
    `{"documents": {stem: scores}, "total": scores}`. The total is a
    micro-average over the summed tp/fp/fn.
  - `main(argv: list[str] | None = None, client: LLMClient | None =
    None) -> int`, with flag `--data` (default `SUBMISSION_DIR / "synth"
    / "data"`). It prints one line per document plus the total. If no
    pairs are found, it prints an error and returns 1.
- Produces, in `optimize/search.py` (stub, raises `NotImplementedError`):
  `search(seed_prompt: str, train: list[tuple[str, str, str]], held_out:
  list[tuple[str, str, str]], client: LLMClient, n: int = 4, rounds: int
  = 3) -> list[tuple[str, float]]`. Fitness is the `score_masked` F1 of a
  candidate's single-shot output. Output whose structure doesn't match
  scores 0.

- [ ] **Step 1: Write the failing tests** (`test_score.py`):
  - `test_perfect_match`: identical maskings → `precision`, `recall` and
    `f1` are all `1.0`.
  - `test_counts_misses_and_extras`: truth `"Nome: * *\nIdade: 30"`,
    predicted `"Nome: * Santos\nIdade: *"` → `tp=1`, `fp=1`, `fn=1`.
  - `test_structure_mismatch_raises`: a different number of tokens on a
    line → `ValueError` with `"line 1"`.
  - `test_nothing_masked_anywhere`: `precision` and `recall` are `1.0`.
  - `test_load_dataset_pairs_files` (temp dir): two pairs plus an items
    file → 2 entries, sorted, with the trailing newline stripped.
  - `test_evaluate_micro_averages`: docs `"Nome: Ana"` / `"Nome: Rui"`,
    both with truth `"Nome: *"`. A fake identity detector returns only
    `Ana`. Expect a total `precision` of `1.0` and `recall` of `0.5`.
  - `test_main_without_data_fails`: an empty directory → returns `1`.
- [ ] **Step 2:** Run `python -m unittest submission.tests.test_score`.
  Expected: FAIL (ImportError).
- [ ] **Step 3:** Implement `evaluate/score.py`, write the `search` stub,
  and write `synth/README.md`. The README covers the file format, the
  command `python -m submission.data_generator --count N`, and the rule
  that labels come from `categories.json`.
- [ ] **Step 4:** Run the tests, the full suite, and `flake8 .`. Expected:
  all OK, flake8 clean.
- [ ] **Step 5:** Update the spec sections listed under Files.
- [ ] **Step 6: Checkpoint.** Suggested message:
  `feat: word-level scoring against synthetic ground truth`.

### Task 12: Final verification

- [ ] **Step 1:** Run `python -m unittest discover -s submission -t .`.
  Expected: every test OK, Rogério's included.
- [ ] **Step 2:** Run `flake8 .`. Expected: no output.
- [ ] **Step 3 (needs the real key, João runs it):** run `python -m
  submission.run --single-shot`, then `python -m submission.run`. Check:
  - the first 4 lines of `submission.txt` match the spec's expected
    header;
  - `outputs/report.md` lists the findings and a meaning score;
  - the unmatched list is empty or explainable.
- [ ] **Step 4: Checkpoint.** João commits `submission.txt`, `prompt.txt`
  and `outputs/`, then opens the PR when the team is ready.

---

## Addendum: consolidated agents and `retryDelay` (after Task 11)

Free-tier keys allow as few as 5 requests per minute. Tasks 8 and 9 built
one detector and one context checker **per category**, running in
parallel: up to 11 calls per document, so a single document was rejected.
This addendum supersedes their interfaces (spec §4 and §6.3 are current).

**Consolidation**
- `detect(text, categories: list[Category], base, client) ->
  list[Finding]`: one call covering every category. Findings keep the
  category the model assigns (a report label only; masking ignores it).
- `check_context(text, categories, findings, base, client) ->
  ContextResult`: one call for all findings, each listed as
  `[category] "text" (linha: "context")`.
- `format_categories(categories) -> str` joins `format_category`, which
  stays unchanged because `data_generator.py` uses it.
- `run_pipeline(text, base, client, categories=None)`: no thread pool and
  no `max_workers`. Calls are sequential: detector, then the context
  checker (skipped when nothing was found), then the reviewer. At most 3.
- `prompts/detector.md` asks the detector to check every category on every
  line; `prompts/context_checker.md` lists the categories and all findings.
- Tests: `test_agents.py` (one call per role, labels kept),
  `test_orchestrator.py` (exactly 3 calls with findings, 2 without).

**`retryDelay`**
- On a retryable 429 (per-minute quota), `GeminiClient` waits the delay
  Google sends (`'retryDelay': '32s'`) instead of 1/2/4 s, capped at
  `MAX_RETRY_DELAY = 60` seconds. Other retryable errors keep the 1/2/4 s
  backoff; a per-day 429 still fails fast.
- Test: `test_429_waits_server_retry_delay` (sleep called with 32).
