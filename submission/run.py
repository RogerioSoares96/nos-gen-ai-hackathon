"""Anonymise the challenge PDF.

Run from the repo root:
    python -m submission.run                 # pipeline -> submission.txt
    python -m submission.run --single-shot   # prompt.txt alone, like CI
"""
import argparse
import sys

from submission.anonymizer import config
from submission.anonymizer.base_prompt import load_base
from submission.anonymizer.extract import pdf_to_text
from submission.anonymizer.llm import LLMClient, make_client
from submission.anonymizer.orchestrator import run_pipeline
from submission.anonymizer.report import write_report, write_utf8
from submission.validation import DocumentValidator


def main(argv: list[str] | None = None,
         client: LLMClient | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--single-shot", action="store_true",
                        help="send prompt.txt alone, as the CI check does")
    parser.add_argument("--no-validate", action="store_true",
                        help="skip the meaning-preservation score")
    parser.add_argument("--no-cache", action="store_true",
                        help="ask Gemini again instead of reusing replies")
    args = parser.parse_args(argv)

    try:
        if client is None:
            client = make_client(config.main_model(),
                                 refresh=args.no_cache)

        if args.single_shot:
            _single_shot(client)
        else:
            _pipeline(client, validate=not args.no_validate)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    return 0


def _single_shot(client: LLMClient) -> None:
    prompt = config.PROMPT_PATH.read_text(encoding="utf-8")
    output = client.generate_text(prompt)
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    write_utf8(config.OUTPUTS_DIR / "single_shot.txt", output)


def _pipeline(client: LLMClient, validate: bool) -> None:
    text = pdf_to_text(config.PDF_PATH)
    result = run_pipeline(text, load_base(), client)

    # Written only after the whole pipeline succeeded, so a failed run
    # never leaves a half-finished submission behind.
    write_utf8(config.SUBMISSION_PATH, result.masked)

    entailment = None
    if validate and config.validation_model():
        try:
            entailment = DocumentValidator().entailment(text, result.masked)
        except Exception as error:
            print(f"warning: validation skipped: {error}", file=sys.stderr)

    write_report(result, config.OUTPUTS_DIR, entailment)


if __name__ == "__main__":
    sys.exit(main())
