import unittest

from submission.anonymizer.agents.context_checker import check_context
from submission.anonymizer.agents.detector import detect
from submission.anonymizer.agents.reviewer import review
from submission.anonymizer.categories import Category
from submission.anonymizer.schemas import Finding
from submission.tests.fakes import FakeLLMClient

HEALTH = Category(
    id="health",
    name="Saúde",
    description="Diagnósticos e medicação.",
    examples=["asma crónica"],
    do_not_mask=["Nomes de secções"],
)
IDENTITY = Category(
    id="identity",
    name="Identidade",
    description="Nomes de pessoas.",
    examples=["Pedro Almeida"],
    do_not_mask=["Papéis sem nome"],
)
CATEGORIES = [IDENTITY, HEALTH]
DOCUMENT = "Nome: Maria Santos\nDiagnóstico: hipertensão"


def finding_json(text, category, context=""):
    return (
        f'{{"text": "{text}", "category": "{category}", '
        f'"reason": "r", "context": "{context}"}}'
    )


class TestDetector(unittest.TestCase):
    def test_detect_is_one_call_for_all_categories(self):
        client = FakeLLMClient(lambda prompt, schema: '{"findings": []}')

        detect(DOCUMENT, CATEGORIES, "BASE", client)

        self.assertEqual(len(client.prompts), 1)
        prompt = client.prompts[0]
        self.assertTrue(prompt.startswith("BASE"))
        for category in CATEGORIES:
            self.assertIn(f"Categoria: {category.id}", prompt)
        self.assertIn("asma crónica", prompt)
        self.assertTrue(prompt.endswith(DOCUMENT))

    def test_detect_keeps_model_category(self):
        reply = (
            f'{{"findings": [{finding_json("Maria Santos", "identity")}, '
            f'{finding_json("hipertensão", "health")}]}}'
        )
        client = FakeLLMClient(lambda prompt, schema: reply)

        findings = detect(DOCUMENT, CATEGORIES, "BASE", client)

        self.assertEqual(
            [(f.text, f.category) for f in findings],
            [("Maria Santos", "identity"), ("hipertensão", "health")],
        )


class TestContextChecker(unittest.TestCase):
    def test_check_context_is_one_call_for_all_findings(self):
        findings = [
            Finding(text="Maria Santos", category="identity", reason="r",
                    context="Nome: Maria Santos"),
            Finding(text="hipertensão", category="health", reason="r",
                    context="Diagnóstico: hipertensão"),
        ]
        reply = (
            f'{{"additions": [{finding_json("Maria", "identity")}], '
            f'"rejections": ["hipertensão"]}}'
        )
        client = FakeLLMClient(lambda prompt, schema: reply)

        result = check_context(DOCUMENT, CATEGORIES, findings, "BASE",
                               client)

        self.assertEqual(len(client.prompts), 1)
        prompt = client.prompts[0]
        for finding in findings:
            self.assertIn(finding.text, prompt)
            self.assertIn(finding.context, prompt)
            self.assertIn(finding.category, prompt)
        self.assertEqual(result.rejections, ["hipertensão"])
        self.assertEqual(result.additions[0].category, "identity")


class TestReviewer(unittest.TestCase):
    def test_review_sends_masked_text(self):
        client = FakeLLMClient(lambda prompt, schema: '{"findings": []}')

        review("Nome: * *", "BASE", client)

        self.assertTrue(client.prompts[0].endswith("Nome: * *"))

    def test_review_findings_keep_their_category(self):
        reply = f'{{"findings": [{finding_json("Católica", "x_social")}]}}'
        client = FakeLLMClient(lambda prompt, schema: reply)

        findings = review("Religião: Católica", "BASE", client)

        self.assertEqual(findings[0].category, "x_social")


if __name__ == "__main__":
    unittest.main()
