import unittest

from submission.anonymizer.schemas import (
    ContextResult,
    EntailmentScore,
    Finding,
)


class TestSchemas(unittest.TestCase):
    def test_finding_context_defaults_to_empty(self):
        finding = Finding(text="Maria", category="identity", reason="name")
        self.assertEqual(finding.context, "")

    def test_entailment_score_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            EntailmentScore.model_validate_json('{"score": 101}')

    def test_context_result_parses_json(self):
        result = ContextResult.model_validate_json(
            '{"additions": [], "rejections": ["Lisboa"]}'
        )
        self.assertEqual(result.rejections, ["Lisboa"])


if __name__ == "__main__":
    unittest.main()
