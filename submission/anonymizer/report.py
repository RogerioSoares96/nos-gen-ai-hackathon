import json
from pathlib import Path

from submission.anonymizer.orchestrator import PipelineResult


def write_report(result: PipelineResult, out_dir: Path,
                 entailment: int | None = None) -> None:
    """Write findings.json (machine-readable) and report.md (for review)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "findings": [f.model_dump() for f in result.findings],
        "rejected": result.rejected,
        "unmatched": [f.model_dump() for f in result.unmatched],
    }
    write_utf8(out_dir / "findings.json",
               json.dumps(data, ensure_ascii=False, indent=2) + "\n")

    score = f"{entailment}/100" if entailment is not None else "not computed"
    lines = [
        "# Anonymisation report",
        "",
        f"Meaning preservation: {score}",
        "",
        f"## Masked findings ({len(result.findings)})",
        "",
        "| Text | Category | Reason |",
        "|---|---|---|",
    ]
    lines += [
        f"| {_cell(f.text)} | {f.category} | {_cell(f.reason)} |"
        for f in result.findings
    ]
    lines += ["", f"## Rejected as not sensitive ({len(result.rejected)})",
              ""]
    lines += [f"- {text}" for text in result.rejected]
    lines += ["", f"## Not found verbatim ({len(result.unmatched)})", ""]
    lines += [f"- {f.text} ({f.category})" for f in result.unmatched]
    write_utf8(out_dir / "report.md", "\n".join(lines) + "\n")


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def write_utf8(path: Path, content: str) -> None:
    # newline="\n" stops Windows from writing CRLF line endings.
    with open(path, "w", encoding="utf-8", newline="\n") as file:
        file.write(content)
