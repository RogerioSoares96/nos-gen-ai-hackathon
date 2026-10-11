"""Score the pipeline against synthetic ground truth, word by word.

Run from the repo root:
    python -m submission.evaluate.score [--data DIR]
"""
import argparse
import sys
from pathlib import Path

from submission.anonymizer import config
from submission.anonymizer.base_prompt import load_base
from submission.anonymizer.llm import LLMClient, make_client
from submission.anonymizer.orchestrator import run_pipeline

DEFAULT_DATA_DIR = config.SUBMISSION_DIR / "synth" / "data"
MASKED_SUFFIX = "_masked"


def score_masked(predicted: str, truth: str) -> dict[str, float]:
    """Precision, recall and F1 over masked words, compared by position.

    Both maskings replace words 1:1 (spec §7), so token positions line up;
    a different structure means the text itself changed and the
    comparison would be meaningless.
    """
    predicted_lines = predicted.split("\n")
    truth_lines = truth.split("\n")
    if len(predicted_lines) != len(truth_lines):
        raise ValueError(
            f"line count differs: {len(predicted_lines)} predicted vs "
            f"{len(truth_lines)} expected"
        )

    tp = fp = fn = 0
    for number, (ours, theirs) in enumerate(
        zip(predicted_lines, truth_lines), start=1
    ):
        our_tokens = ours.split()
        their_tokens = theirs.split()
        if len(our_tokens) != len(their_tokens):
            raise ValueError(f"token count differs on line {number}")

        for our_token, their_token in zip(our_tokens, their_tokens):
            ours_masked = "*" in our_token
            theirs_masked = "*" in their_token
            tp += ours_masked and theirs_masked
            fp += ours_masked and not theirs_masked
            fn += theirs_masked and not ours_masked

    return _scores(tp, fp, fn)


def load_dataset(data_dir: Path) -> list[tuple[str, str, str]]:
    """(stem, text, masked) for every document with a masked twin."""
    dataset = []
    for masked_path in sorted(data_dir.glob(f"*{MASKED_SUFFIX}.txt")):
        stem = masked_path.stem[:-len(MASKED_SUFFIX)]
        text_path = data_dir / f"{stem}.txt"
        if not text_path.exists():
            continue

        dataset.append((stem, _read(text_path), _read(masked_path)))
    return dataset


def evaluate(dataset: list[tuple[str, str, str]], base: str,
             client: LLMClient) -> dict:
    documents = {}
    for stem, text, truth in dataset:
        result = run_pipeline(text, base, client)
        documents[stem] = score_masked(result.masked, truth)

    total = _scores(*(
        sum(scores[key] for scores in documents.values())
        for key in ("tp", "fp", "fn")
    ))
    return {"documents": documents, "total": total}


def main(argv: list[str] | None = None,
         client: LLMClient | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR,
                        help="directory with *.txt / *_masked.txt pairs")
    parser.add_argument("--no-cache", action="store_true",
                        help="ask Gemini again instead of reusing replies")
    args = parser.parse_args(argv)

    dataset = load_dataset(args.data)
    if not dataset:
        print(f"error: no document pairs in {args.data}", file=sys.stderr)
        return 1

    try:
        if client is None:
            client = make_client(config.main_model(),
                                 refresh=args.no_cache)
        result = evaluate(dataset, load_base(), client)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    for stem, scores in result["documents"].items():
        print(f"{stem}: {_format(scores)}")
    print(f"TOTAL: {_format(result['total'])}")
    return 0


def _scores(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def _format(scores: dict[str, float]) -> str:
    return (f"P={scores['precision']:.3f} R={scores['recall']:.3f} "
            f"F1={scores['f1']:.3f} (tp={scores['tp']} fp={scores['fp']} "
            f"fn={scores['fn']})")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").removesuffix("\n")


if __name__ == "__main__":
    sys.exit(main())
