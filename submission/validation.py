import re

from submission.anonymizer import config
from submission.anonymizer.llm import GeminiClient, LLMClient
from submission.anonymizer.schemas import EntailmentScore


class DocumentValidator:
    """Validate meaning preservation and the number of masked words."""

    def __init__(self, client: LLMClient | None = None):
        self._client = client

    def entailment(self, original_document: str, last_document: str) -> int:
        """Score how well the anonymized document preserves the original."""
        prompt = f"""
Compare the original document with the anonymized document below.
Score from 0 to 100 how well the anonymized document preserves the original
document's context, meaning, and structure. A score of 100 means the content
is fully preserved apart from appropriate anonymization. A score of 0 means
the content is not preserved. Masked sensitive words represented by one or
more asterisks are expected and should not reduce the score by themselves.

Original document:
{original_document}

Anonymized document:
{last_document}
""".strip()

        result = self._get_client().generate_json(prompt, EntailmentScore)
        return result.score

    def check_removed_words(self, document: str, expected_count: int) -> bool:
        """Check whether the document has the expected number of masks."""
        if expected_count < 0:
            raise ValueError("expected_count must not be negative")

        masked_words = re.findall(r"\*+", document)
        return len(masked_words) == expected_count

    def _get_client(self) -> LLMClient:
        if self._client is None:
            validation_model = config.validation_model()
            if not validation_model:
                raise RuntimeError(
                    "VALIDATION_MODEL environment variable is not set"
                )
            self._client = GeminiClient(model=validation_model)
        return self._client
