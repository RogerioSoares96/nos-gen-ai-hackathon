# Synthetic data (owner: Salvador)

Labelled documents that look like `raw_data/document_to_anonymize.pdf`.
They exist to **measure** the pipeline (`evaluate/score.py`) and to
**optimise** `prompt.txt` (`optimize/search.py`). They are never part of
the submission run.

## Generating

From the repo root:

```
python -m submission.data_generator --count 5
```

By default, files land in `submission/synth/data/`. Use `--outdir` to
change it, and `--write-items-file` to also list the labelled spans.

## Files

For each document `NNN`:

| File | Content |
|---|---|
| `synthetic_document_NNN.txt` | the document, without empty lines |
| `synthetic_document_NNN_masked.txt` | the ground truth (masked) |
| `synthetic_document_NNN_sensitive_to_mask.txt` | optional: `category \| text \| context` |

Each file ends with a single `\n`.

## Why the ground truth is comparable

- **Same categories.** The generator's prompt lists the categories from
  `anonymizer/categories.json`, so what it labels as sensitive is exactly
  what the pipeline looks for. Changing the taxonomy there changes both.
- **Same masker.** The masked files come from the pipeline's own
  `apply_findings`: one `*` per word, edge punctuation kept, whole-word
  matches, limited to each span's `context` line (spec §7).

Because of this, `evaluate/score.py` can compare our output with the
ground truth word by word, the same way the hidden CI diffs
`submission.txt`.

## Scoring

```
python -m submission.evaluate.score
```

This runs the pipeline on every pair and prints precision, recall and F1
over masked words, per document and in total. Keep a few documents aside
as a held-out set for `optimize/` (spec §11).
