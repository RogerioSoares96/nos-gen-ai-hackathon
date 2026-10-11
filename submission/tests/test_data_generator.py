import os
import unittest
from unittest.mock import patch

from submission import data_generator
from submission.anonymizer import config
from submission.anonymizer.categories import load_categories


def item(text, category="identity", context=""):
    return {"category": category, "text": text, "context": context}


class TestMaskDocument(unittest.TestCase):
    def test_uses_project_masking_rule(self):
        document = ("Data: 12/03/1978\nEmail: ana.c@x.pt\n"
                    "Telefone: +351 912 345 678\nCC: 12345678-9ZX0")
        items = [item("12/03/1978"), item("ana.c@x.pt"),
                 item("+351 912 345 678"), item("12345678-9ZX0")]

        masked = data_generator.mask_document(document, items)

        self.assertEqual(
            masked, "Data: *\nEmail: *\nTelefone: * * * *\nCC: *"
        )

    def test_no_substring_match(self):
        masked = data_generator.mask_document("Ana fez Anamnese",
                                              [item("Ana")])
        self.assertEqual(masked, "* fez Anamnese")

    def test_context_limits_masking(self):
        document = ("Relatório - Centro Médico Lisboa\n"
                    "Morada: Rua do Sol, Lisboa")
        items = [item("Lisboa", context="Morada: Rua do Sol, Lisboa")]

        masked = data_generator.mask_document(document, items)

        self.assertEqual(masked, "Relatório - Centro Médico Lisboa\n"
                                 "Morada: Rua do Sol, *")


class TestPayload(unittest.TestCase):
    def test_normalize_keeps_context(self):
        payload = {"document_text": "Nome: Ana\n\nIdade: 30",
                   "sensitive_items": [item("Ana", context="Nome: Ana")]}

        normalized = data_generator.normalize_payload(payload)

        self.assertEqual(normalized["document_text"], "Nome: Ana\nIdade: 30")
        self.assertEqual(normalized["sensitive_items"][0]["context"],
                         "Nome: Ana")


class TestBuildPrompt(unittest.TestCase):
    def setUp(self):
        self.prompt = data_generator.build_prompt(1)

    def test_lists_every_project_category(self):
        for category in load_categories():
            self.assertIn(category.id, self.prompt)
            self.assertIn(category.description, self.prompt)

    def test_health_data_is_sensitive(self):
        # The project masks clinical data (spec §8); the generator's
        # ground truth must agree, or every health finding scores as a
        # false positive.
        for phrase in ("Diagnosis names", "Medication names", "Symptoms"):
            self.assertNotIn(phrase, self.prompt)


class TestDefaults(unittest.TestCase):
    def test_model_comes_from_data_gen_setting(self):
        args = data_generator.parse_args(["--count", "1"])
        with (
            patch.dict(os.environ, {"DATA_GEN_MODEL": "gen"}, clear=True),
            patch("submission.anonymizer.config.load_dotenv"),
        ):
            self.assertEqual(data_generator.resolve_model(args), "gen")

    def test_model_flag_overrides_setting(self):
        args = data_generator.parse_args(["--count", "1", "--model", "m"])
        self.assertEqual(data_generator.resolve_model(args), "m")

    def test_outdir_defaults_to_synth_data(self):
        args = data_generator.parse_args(["--count", "1"])
        self.assertEqual(args.outdir,
                         str(config.SUBMISSION_DIR / "synth" / "data"))


if __name__ == "__main__":
    unittest.main()
