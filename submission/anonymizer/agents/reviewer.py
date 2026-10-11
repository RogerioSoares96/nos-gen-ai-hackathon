from submission.anonymizer.base_prompt import compose
from submission.anonymizer.llm import LLMClient
from submission.anonymizer.schemas import Finding, ReviewResult


def review(masked_text: str, base: str, client: LLMClient) -> list[Finding]:
    prompt = compose(base, "reviewer", masked_text)
    return client.generate_json(prompt, ReviewResult).findings
