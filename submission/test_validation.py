import os
import re
import unittest
from unittest.mock import patch

from submission.tests.fakes import FakeLLMClient
from submission.validation import DocumentValidator


class TestValidation(unittest.TestCase):
    def test_default_client_uses_validation_model(self):
        with (
            patch.dict(os.environ, {"VALIDATION_MODEL": "models/test-model"}),
            patch("submission.anonymizer.config.load_dotenv"),
            patch("submission.validation.GeminiClient") as client_cls,
        ):
            client_cls.return_value.generate_json.return_value.score = 90
            score = DocumentValidator().entailment("Original", "Anonymized")

        self.assertEqual(score, 90)
        print(f"Default client model: {client_cls.call_args}", flush=True)
        client_cls.assert_called_once_with(model="models/test-model")

    def test_missing_validation_model_raises(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("submission.anonymizer.config.load_dotenv"),
        ):
            with self.assertRaises(RuntimeError):
                DocumentValidator().entailment("Original", "Anonymized")

    def test_entailment_returns_score(self):
        original_document = """Relatório de Admissão
Nome: Maria Santos
Telefone: +351 912 345 678
Diagnóstico: hipertensão"""
        anonymized_document = """Relatório de Admissão
Nome: * *
Telefone: *
Diagnóstico: hipertensão"""
        client = FakeLLMClient(lambda prompt, schema: '{"score": 87}')

        score = DocumentValidator(client).entailment(
            original_document,
            anonymized_document,
        )

        self.assertIsInstance(score, int)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
        print("\n=== Entailment validation ===", flush=True)
        print("Original document:", flush=True)
        print(original_document, flush=True)
        print("\nAnonymized document:", flush=True)
        print(anonymized_document, flush=True)
        print(f"\nAPI response: {{'score': {score}}}", flush=True)
        print(f"Entailment score: {score}/100", flush=True)
        print("Entailment score valid: yes", flush=True)
        self.assertEqual(len(client.prompts), 1)
        prompt = client.prompts[0]
        self.assertIn(original_document, prompt)
        self.assertIn(anonymized_document, prompt)

    def test_entailment_rejects_scores_outside_range(self):
        client = FakeLLMClient(lambda prompt, schema: '{"score": 101}')

        with self.assertRaises(ValueError):
            DocumentValidator(client).entailment("Original", "Anonymized")

    def test_check_removed_words_counts_mask_sequences(self):
        validator = DocumentValidator()
        anonymized_document = """Nome: * *
Telefone: *
Email: *"""

        expected_count = 4
        removed_words = re.findall(r"\*+", anonymized_document)
        removed_count = len(removed_words)
        result = validator.check_removed_words(
            anonymized_document, expected_count
        )
        print("\n=== Removed-word validation ===", flush=True)
        print("Anonymized document:", flush=True)
        print(anonymized_document, flush=True)
        print(f"Masked values found: {removed_words}", flush=True)
        print(f"Removed words: {removed_count}", flush=True)
        print(f"Expected removed words: {expected_count}", flush=True)
        print(
            f"Word count matches: {'yes' if result else 'no'}", flush=True
        )
        self.assertEqual(removed_count, expected_count)
        self.assertTrue(result)

    def test_check_removed_words_rejects_negative_expected_count(self):
        with self.assertRaises(ValueError):
            DocumentValidator().check_removed_words("Name: *", -1)


if __name__ == "__main__":
    unittest.main()
