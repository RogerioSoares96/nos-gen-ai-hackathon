#!/usr/bin/env python3
"""
Generate synthetic medical admission documents and masked versions using
Gemini.

What counts as sensitive comes from anonymizer/categories.json, and masking
uses the pipeline's own apply_findings, so the ground truth follows the same
rules as submission.txt.

.env file:
    Either of these is accepted:
        GEMINI_API_KEY=your_gemini_api_key_here
    or:
        API_KEY=your_gemini_api_key_here

Examples (from the repo root):
    python -m submission.data_generator --count 2
    python -m submission.data_generator --count 5 --outdir generated
    python -m submission.data_generator --count 2 --max-retries 6
    python -m submission.data_generator --count 2 --write-items-file
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types

from submission.anonymizer import config
from submission.anonymizer.agents.detector import format_category
from submission.anonymizer.categories import load_categories
from submission.anonymizer.masking import apply_findings
from submission.anonymizer.schemas import Finding

DEFAULT_OUTDIR = config.SUBMISSION_DIR / "synth" / "data"

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def build_prompt(document_number: int) -> str:
    categories = "\n\n".join(
        format_category(category) for category in load_categories()
    )
    return f"""
Generate one fully synthetic Portuguese medical admission report.

The document must be similar in style, structure, and level of detail to a
hospital or medical admission report from Portugal.

The document must be written in Portuguese from Portugal.

All data must be completely fictitious.
Do not use real people, real patients, real public figures, or real
addresses tied to real individuals.
Use realistic but mock data only.

The response must be valid JSON only.
Do not include markdown.
Do not include explanations outside the JSON.

Return exactly this JSON structure:

{{
  "document_text": "Full synthetic report text here",
  "sensitive_items": [
    {{
      "category": "identity",
      "text": "Exact sensitive text span appearing in document_text",
      "context": "The full line of document_text where the span appears"
    }}
  ]
}}

The generated document should include realistic sections such as:
- Relatório de Admissão
- Data
- Referência
- Informações do Paciente
- Histórico Médico
- Informações Sociais e Comportamentais
- Informações Financeiras
- Dados Biométricos
- Contactos de Emergência
- Assinatura Digital

Sensitive information, by category. Every span of these categories MUST be
included in sensitive_items, and "category" MUST be one of these ids:

{categories}

Information that MUST NOT be included in sensitive_items:
- Document labels, such as "Nome:", "Data de Nascimento:", "Morada:"
- Section headers, such as "Informações do Paciente:"
- The institution name in the report title
- The document creation date, such as "Data: 15 de abril de 2025"
- The report reference code, such as "Referência: ADM-2025-04-15-089"

Rules for sensitive_items:
- "text" and "context" are exact substrings of document_text.
- "context" is the whole line that contains "text".
- Include minimal spans that should be masked while preserving the
  sentence context.
- Do not include full sentences unless the entire sentence is sensitive.
- Do not include labels or punctuation that can safely remain visible.
- If a value appears on several lines, include one item per line.
- Avoid duplicates.
- Be comprehensive.

Masking examples:
- "Ana Correia" should be listed as "Ana Correia"
- "Rua das Flores, 123, 2.º Esq., 1000-001 Lisboa" should be listed as
  that exact address
- "ana.correia@example.invalid" should be listed as that exact email

Document number: {document_number}
"""


def remove_empty_lines(text: str) -> str:
    """
    Remove empty lines while preserving all non-empty line content exactly.
    """
    lines = text.splitlines()
    non_empty_lines = [line for line in lines if line.strip() != ""]
    return "\n".join(non_empty_lines)


def extract_json(text: str) -> Dict[str, Any]:
    """
    Parse Gemini's JSON response.

    If the model accidentally wraps JSON in markdown fences, remove them.
    """
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE
        ).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Could not parse Gemini response as JSON.\n\n"
            f"Response was:\n{cleaned}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError("Gemini response is not a JSON object.")

    return payload


def normalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean document_text and sensitive_items.
    """
    document_text = payload.get("document_text", "")
    sensitive_items = payload.get("sensitive_items", [])

    if isinstance(document_text, str):
        payload["document_text"] = remove_empty_lines(document_text.strip())

    if isinstance(sensitive_items, list):
        cleaned_items = []

        for item in sensitive_items:
            if not isinstance(item, dict):
                continue

            category = str(item.get("category", "")).strip()
            text = str(item.get("text", "")).strip()
            context = str(item.get("context", "")).strip()

            if not category or not text:
                continue

            cleaned_items.append(
                {
                    "category": category,
                    "text": text,
                    "context": context,
                }
            )

        payload["sensitive_items"] = cleaned_items

    return payload


def validate_payload(payload: Dict[str, Any]) -> None:
    """
    Basic validation to make sure the generated response has the expected
    shape.
    """
    if not isinstance(payload, dict):
        raise ValueError("Gemini response is not a JSON object.")

    if "document_text" not in payload:
        raise ValueError("Missing field: document_text.")

    if "sensitive_items" not in payload:
        raise ValueError("Missing field: sensitive_items.")

    if not isinstance(payload["document_text"], str):
        raise ValueError("document_text must be a string.")

    if not isinstance(payload["sensitive_items"], list):
        raise ValueError("sensitive_items must be a list.")

    document_text = payload["document_text"]
    missing_items: List[str] = []

    for item in payload["sensitive_items"]:
        if not isinstance(item, dict):
            raise ValueError("Each sensitive item must be an object.")

        if "category" not in item or "text" not in item:
            raise ValueError(
                "Each sensitive item must contain category and text."
            )

        if not isinstance(item["category"], str):
            raise ValueError("Sensitive item category must be a string.")

        if not isinstance(item["text"], str):
            raise ValueError("Sensitive item text must be a string.")

        text = item["text"].strip()

        if text and text not in document_text:
            missing_items.append(text)

    if missing_items:
        print(
            "Warning: Some sensitive items were not found exactly in the "
            "document text:",
            file=sys.stderr,
        )
        for value in missing_items:
            print(f"  - {value}", file=sys.stderr)


def get_status_code_from_exception(exc: Exception) -> Optional[int]:
    """
    Try to extract an HTTP status code from google-genai exceptions.
    """
    status_code = getattr(exc, "status_code", None)

    if isinstance(status_code, int):
        return status_code

    match = re.search(r"\b(429|500|502|503|504)\b", str(exc))

    if match:
        return int(match.group(1))

    return None


def is_retryable_error(exc: Exception) -> bool:
    """
    Return True for temporary API errors worth retrying.
    """
    status_code = get_status_code_from_exception(exc)

    if status_code in RETRYABLE_STATUS_CODES:
        return True

    text = str(exc).lower()

    retryable_markers = [
        "unavailable",
        "high demand",
        "overloaded",
        "rate limit",
        "quota",
        "temporarily",
        "timeout",
        "connection",
    ]

    return any(marker in text for marker in retryable_markers)


def generate_document(
    client: genai.Client,
    model: str,
    document_number: int,
    max_retries: int,
) -> Dict[str, Any]:
    """
    Generate one document with retries for temporary Gemini/API failures.
    """
    prompt = build_prompt(document_number)
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.9,
                    top_p=0.95,
                    response_mime_type="application/json",
                ),
            )

            if not response.text:
                raise RuntimeError("Gemini returned an empty response.")

            payload = extract_json(response.text)
            payload = normalize_payload(payload)
            validate_payload(payload)

            return payload

        except Exception as exc:
            last_error = exc

            if not is_retryable_error(exc):
                raise

            if attempt >= max_retries:
                break

            backoff_seconds = min(
                30.0,
                float(2 ** (attempt - 1)) + random.uniform(0.25, 1.25),
            )

            print(
                "Temporary Gemini/API error while generating document "
                f"{document_number}. Attempt {attempt}/{max_retries}. "
                f"Reason: {exc}",
                file=sys.stderr,
            )

            time.sleep(backoff_seconds)

    raise RuntimeError(
        f"Failed to generate document {document_number} after "
        f"{max_retries} attempts. Last error: {last_error}"
    )


def build_unique_sensitive_items(
    sensitive_items: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """
    Remove duplicate sensitive items.
    """
    seen: Set[Tuple[str, str, str]] = set()
    unique_items: List[Dict[str, str]] = []

    for item in sensitive_items:
        category = item.get("category", "").strip()
        text = item.get("text", "").strip()
        context = item.get("context", "").strip()

        if not category or not text:
            continue

        key = (category.lower(), text, context)

        if key in seen:
            continue

        seen.add(key)
        unique_items.append(
            {
                "category": category,
                "text": text,
                "context": context,
            }
        )

    return unique_items


def mask_document(
    document_text: str,
    sensitive_items: List[Dict[str, str]],
) -> str:
    """
    Mask the sensitive items with the pipeline's own masker, so the ground
    truth follows exactly the rules submission.txt is produced with
    (spec §7): one '*' per whitespace-separated word, edge punctuation
    kept, whole-word matches only, and masking limited to each item's
    context line when it has one.

    Removes empty lines.
    """
    findings = [
        Finding(
            text=item["text"],
            category=item["category"],
            reason="synthetic ground truth",
            context=item.get("context", ""),
        )
        for item in build_unique_sensitive_items(sensitive_items)
    ]
    result = apply_findings(remove_empty_lines(document_text), findings)
    return result.masked


def write_outputs(
    payload: Dict[str, Any],
    outdir: Path,
    document_number: int,
    write_items_file: bool,
) -> None:
    """
    Writes:
    - Main synthetic document .txt
    - Masked synthetic document .txt
    - Optional sensitive items list .txt
    """
    outdir.mkdir(parents=True, exist_ok=True)

    stem = f"synthetic_document_{document_number:03d}"
    document_path = outdir / f"{stem}.txt"
    masked_path = outdir / f"{stem}_masked.txt"
    items_path = outdir / f"{stem}_sensitive_to_mask.txt"

    document_text = remove_empty_lines(payload["document_text"].strip())
    sensitive_items = build_unique_sensitive_items(payload["sensitive_items"])
    masked_text = mask_document(document_text, sensitive_items)

    with document_path.open("w", encoding="utf-8", newline="\n") as file:
        file.write(document_text)
        file.write("\n")

    with masked_path.open("w", encoding="utf-8", newline="\n") as file:
        file.write(masked_text)
        file.write("\n")

    print(f"Created: {document_path}")
    print(f"Created: {masked_path}")

    if write_items_file:
        with items_path.open("w", encoding="utf-8", newline="\n") as file:
            file.write("Sensitive information to mask\n")
            file.write("=============================\n")

            for item in sensitive_items:
                file.write(
                    f"{item['category']} | {item['text']} | "
                    f"{item['context']}\n"
                )

        print(f"Created: {items_path}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic sensitive documents using Gemini."
    )

    parser.add_argument(
        "--count",
        type=int,
        required=True,
        help="Number of synthetic documents to generate.",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default=str(DEFAULT_OUTDIR),
        help="Directory for the generated .txt files. "
             f"Default: {DEFAULT_OUTDIR}",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Gemini model name. Default: env DATA_GEN_MODEL, "
             "then MAIN_MODEL.",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retry attempts per document for temporary API "
             "errors. Default: 5",
    )

    parser.add_argument(
        "--write-items-file",
        action="store_true",
        help="Also write a .txt file listing the sensitive spans used for "
             "masking.",
    )

    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop the whole script if one document fails. By default, "
             "failed documents are skipped.",
    )

    return parser.parse_args(argv)


def resolve_model(args: argparse.Namespace) -> str:
    return args.model or config.data_gen_model()


def get_api_key() -> str:
    """
    Read Gemini API key from environment.

    Accepts both GEMINI_API_KEY and API_KEY so your current .env keeps
    working.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")

    if not api_key:
        raise EnvironmentError(
            "Missing API key. Add one of these to your .env file:\n"
            "GEMINI_API_KEY=your_gemini_api_key_here\n"
            "or\n"
            "API_KEY=your_gemini_api_key_here"
        )

    return api_key


def main() -> None:
    load_dotenv()

    args = parse_args()

    if args.count < 1:
        raise ValueError("--count must be at least 1.")

    if args.max_retries < 1:
        raise ValueError("--max-retries must be at least 1.")

    api_key = get_api_key()
    client = genai.Client(api_key=api_key)
    outdir = Path(args.outdir)
    model = resolve_model(args)

    successful = 0
    failed = 0

    for document_number in range(1, args.count + 1):
        try:
            payload = generate_document(
                client=client,
                model=model,
                document_number=document_number,
                max_retries=args.max_retries,
            )

            write_outputs(
                payload=payload,
                outdir=outdir,
                document_number=document_number,
                write_items_file=args.write_items_file,
            )

            successful += 1

        except Exception as exc:
            failed += 1

            print(
                f"Error: failed to generate document {document_number}. "
                f"{exc}",
                file=sys.stderr,
            )

            if args.fail_fast:
                raise

    print()
    print("Generation summary")
    print("==================")
    print(f"Requested documents: {args.count}")
    print(f"Successfully generated: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {outdir}")


if __name__ == "__main__":
    main()
