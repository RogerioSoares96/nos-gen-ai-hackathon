from dataclasses import dataclass

from submission.anonymizer.agents.context_checker import check_context
from submission.anonymizer.agents.detector import detect
from submission.anonymizer.agents.reviewer import review
from submission.anonymizer.categories import Category, load_categories
from submission.anonymizer.llm import LLMClient
from submission.anonymizer.masking import apply_findings
from submission.anonymizer.schemas import Finding


@dataclass
class PipelineResult:
    masked: str
    findings: list[Finding]
    rejected: list[str]
    unmatched: list[Finding]


def run_pipeline(text: str, base: str, client: LLMClient,
                 categories: list[Category] | None = None
                 ) -> PipelineResult:
    """Detect -> context-check -> mask -> review -> mask again.

    At most 3 Gemini calls, made one after another: free-tier keys allow as
    few as 5 requests per minute, so a burst of parallel per-category calls
    would be rejected before the document is done.
    """
    if categories is None:
        categories = load_categories()

    findings = detect(text, categories, base, client)

    rejected = []
    if findings:
        check = check_context(text, categories, findings, base, client)
        findings = findings + check.additions
        rejected = check.rejections

    findings = _dedupe(f for f in findings if f.text not in rejected)
    first_pass = apply_findings(text, findings)

    # The reviewer reads the masked text, so its findings (and their
    # context lines) are applied to that text, not to the original.
    reviewed = _dedupe(review(first_pass.masked, base, client))
    second_pass = apply_findings(first_pass.masked, reviewed)

    return PipelineResult(
        masked=second_pass.masked,
        findings=findings + reviewed,
        rejected=rejected,
        unmatched=first_pass.unmatched + second_pass.unmatched,
    )


def _dedupe(findings) -> list[Finding]:
    unique = {}
    for finding in findings:
        unique.setdefault((finding.text, finding.context), finding)
    return list(unique.values())
