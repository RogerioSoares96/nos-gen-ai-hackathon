import re
from dataclasses import dataclass

from submission.anonymizer.schemas import Finding

EDGE_PUNCTUATION = ",.;:()!?"


@dataclass
class MaskResult:
    masked: str
    unmatched: list[Finding]


def apply_findings(text: str, findings: list[Finding]) -> MaskResult:
    """Replace every word overlapping a finding with a single '*'."""
    masked_chars = set()
    unmatched = []

    for finding in findings:
        if not finding.text.strip():
            continue

        spans = _find_spans(text, finding)
        if not spans:
            unmatched.append(finding)
            continue

        for start, end in spans:
            masked_chars.update(range(start, end))

    return MaskResult(_mask_words(text, masked_chars), unmatched)


def _find_spans(text: str, finding: Finding) -> list[tuple[int, int]]:
    if finding.context:
        regions = [
            (m.start(), m.end())
            for m in re.finditer(re.escape(finding.context), text)
        ]
    else:
        regions = [(0, len(text))]

    # Word boundaries stop "Ana" from matching inside "Anamnese".
    pattern = re.compile(r"(?<!\w)" + re.escape(finding.text) + r"(?!\w)")
    spans = []
    for start, end in regions:
        for match in pattern.finditer(text, start, end):
            spans.append((match.start(), match.end()))
    return spans


def _mask_words(text: str, masked_chars: set[int]) -> str:
    lines = []
    offset = 0

    for line in text.split("\n"):
        pieces = []
        last = 0
        for token in re.finditer(r"\S+", line):
            start = offset + token.start()
            end = offset + token.end()
            pieces.append(line[last:token.start()])

            if masked_chars.intersection(range(start, end)):
                pieces.append(_mask_token(token.group()))
            else:
                pieces.append(token.group())

            last = token.end()

        pieces.append(line[last:])
        lines.append("".join(pieces))
        offset += len(line) + 1

    return "\n".join(lines)


def _mask_token(token: str) -> str:
    core = token.strip(EDGE_PUNCTUATION)
    if not core:
        return token

    lead = token[:len(token) - len(token.lstrip(EDGE_PUNCTUATION))]
    trail = token[len(token.rstrip(EDGE_PUNCTUATION)):]
    return lead + "*" + trail
