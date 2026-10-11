import tempfile
import unittest
from pathlib import Path

from submission.anonymizer.base_prompt import MARKER, compose, load_base
from submission.anonymizer.categories import load_categories
from submission.anonymizer.config import PROMPT_PATH


class TestLoadBase(unittest.TestCase):
    def write_prompt(self, content):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "prompt.txt"
        path.write_text(content, encoding="utf-8")
        return path

    def test_load_base_drops_document(self):
        path = self.write_prompt("Instr\n### DOCUMENTO ###\nRelatório")
        self.assertEqual(load_base(path), "Instr")

    def test_load_base_ignores_marker_mentioned_inline(self):
        path = self.write_prompt(
            'Lê o texto após "### DOCUMENTO ###".\nRegras\n'
            "### DOCUMENTO ###\nRelatório"
        )
        self.assertEqual(
            load_base(path),
            'Lê o texto após "### DOCUMENTO ###".\nRegras',
        )

    def test_load_base_without_marker_returns_all(self):
        path = self.write_prompt("Instr only\n")
        self.assertEqual(load_base(path), "Instr only")


class TestCompose(unittest.TestCase):
    def test_compose_orders_parts(self):
        prompt = compose("BASE", "detector", "DOC", category_block="CAT")
        self.assertTrue(prompt.startswith("BASE"))
        self.assertIn("CAT", prompt)
        self.assertTrue(prompt.endswith("### DOCUMENTO ###\nDOC"))

    def test_every_template_composes(self):
        fields = {"category_block": "CAT", "findings_block": "FIND"}
        for agent in ("detector", "context_checker", "reviewer"):
            prompt = compose("BASE", agent, "DOC", **fields)
            self.assertNotIn("$", prompt)


class TestPromptFile(unittest.TestCase):
    def setUp(self):
        self.text = PROMPT_PATH.read_text(encoding="utf-8")

    def test_prompt_txt_mentions_every_category(self):
        for category in load_categories():
            self.assertIn(category.id, self.text)

    def test_prompt_txt_ends_with_document(self):
        document = self.text.split(f"\n{MARKER}\n", 1)[1]
        self.assertTrue(document.startswith(
            "Relatório de Admissão - Centro Médico Lisboa"
        ))


if __name__ == "__main__":
    unittest.main()
