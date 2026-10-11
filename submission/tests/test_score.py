import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from submission.anonymizer import config
from submission.evaluate import score
from submission.evaluate.score import (
    evaluate,
    load_dataset,
    main,
    score_masked,
)
from submission.tests.fakes import FakeLLMClient


def no_findings(prompt, schema):
    if "verificação de contexto" in prompt:
        return json.dumps({"additions": [], "rejections": []})
    return json.dumps({"findings": []})


class TestScoreMasked(unittest.TestCase):
    def test_perfect_match(self):
        scores = score_masked("Nome: * *", "Nome: * *")
        self.assertEqual(
            (scores["precision"], scores["recall"], scores["f1"]),
            (1.0, 1.0, 1.0),
        )

    def test_counts_misses_and_extras(self):
        scores = score_masked("Nome: * Santos\nIdade: *",
                              "Nome: * *\nIdade: 30")
        self.assertEqual((scores["tp"], scores["fp"], scores["fn"]),
                         (1, 1, 1))
        self.assertEqual(scores["precision"], 0.5)
        self.assertEqual(scores["recall"], 0.5)

    def test_structure_mismatch_raises(self):
        with self.assertRaises(ValueError) as ctx:
            score_masked("Nome: *", "Nome: * *")
        self.assertIn("line 1", str(ctx.exception))

    def test_nothing_masked_anywhere(self):
        scores = score_masked("Sexo: Feminino", "Sexo: Feminino")
        self.assertEqual((scores["precision"], scores["recall"]),
                         (1.0, 1.0))


class TestDataset(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.data = Path(directory.name)

    def write(self, name, content):
        (self.data / name).write_text(content + "\n", encoding="utf-8")

    def test_load_dataset_pairs_files(self):
        self.write("synthetic_document_002.txt", "Nome: Rui")
        self.write("synthetic_document_002_masked.txt", "Nome: *")
        self.write("synthetic_document_001.txt", "Nome: Ana")
        self.write("synthetic_document_001_masked.txt", "Nome: *")
        self.write("synthetic_document_001_sensitive_to_mask.txt", "x")

        dataset = load_dataset(self.data)

        self.assertEqual(dataset, [
            ("synthetic_document_001", "Nome: Ana", "Nome: *"),
            ("synthetic_document_002", "Nome: Rui", "Nome: *"),
        ])

    def test_main_without_data_fails(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = main(["--data", str(self.data)],
                        client=FakeLLMClient(None))
        self.assertEqual(code, 1)

    def test_replies_are_cached_unless_no_cache(self):
        self.write("synthetic_document_001.txt", "Nome: Ana")
        self.write("synthetic_document_001_masked.txt", "Nome: *")

        for extra, refresh in (([], False), (["--no-cache"], True)):
            with self.subTest(extra=extra), \
                    patch.object(config, "main_model", return_value="m"), \
                    patch.object(score, "make_client") as make_client, \
                    contextlib.redirect_stdout(io.StringIO()):
                make_client.return_value = FakeLLMClient(no_findings)

                code = main(["--data", str(self.data)] + extra)

                self.assertEqual(code, 0)
                make_client.assert_called_once_with("m", refresh=refresh)


class TestEvaluate(unittest.TestCase):
    def test_evaluate_micro_averages(self):
        def respond(prompt, schema):
            if ("detetor de dados sensíveis" in prompt
                    and prompt.endswith("Nome: Ana")):
                return json.dumps({"findings": [{
                    "text": "Ana", "category": "identity",
                    "reason": "r", "context": "",
                }]})
            if "verificação de contexto" in prompt:
                return json.dumps({"additions": [], "rejections": []})
            return json.dumps({"findings": []})

        dataset = [("doc1", "Nome: Ana", "Nome: *"),
                   ("doc2", "Nome: Rui", "Nome: *")]

        result = evaluate(dataset, "BASE", FakeLLMClient(respond))

        self.assertEqual(result["documents"]["doc1"]["recall"], 1.0)
        self.assertEqual(result["documents"]["doc2"]["recall"], 0.0)
        self.assertEqual(result["total"]["precision"], 1.0)
        self.assertEqual(result["total"]["recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
