from pathlib import Path

import pymupdf


def pdf_to_text(path: str | Path) -> str:
    """Extract the raw text of every page, without empty lines.

    Accents and punctuation are kept as-is: the output contract only
    allows removing empty lines, so no other normalisation happens here.
    """
    with pymupdf.open(path) as document:
        raw = "".join(page.get_text() for page in document)
    return clean_lines(raw)


def clean_lines(raw: str) -> str:
    lines = (line.rstrip() for line in raw.splitlines())
    return "\n".join(line for line in lines if line)
