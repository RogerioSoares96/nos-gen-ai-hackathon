import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from submission import run
from submission.anonymizer import config
from submission.anonymizer.extract import pdf_to_text
from submission.tests.fakes import FakeLLMClient


def no_findings(prompt, schema):
    if "verificação de contexto" in prompt:
        return json.dumps({"additions": [], "rejections": []})
    return json.dumps({"findings": []})


class TestMain(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.submission = root / "submission.txt"
        self.submission.write_text("OLD", encoding="utf-8")
        self.outputs = root / "outputs"

        for name, value in (("SUBMISSION_PATH", self.submission),
                            ("OUTPUTS_DIR", self.outputs)):
            patcher = patch.object(config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_main(self, args, client):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = run.main(args, client=client)
        return code, stderr.getvalue()

    def test_main_writes_submission_only_on_success(self):
        code, _ = self.run_main(["--no-validate"],
                                FakeLLMClient(no_findings))

        self.assertEqual(code, 0)
        written = self.submission.read_bytes()
        self.assertNotIn(b"\r", written)
        self.assertEqual(written.decode("utf-8"),
                         pdf_to_text(config.PDF_PATH))
        self.assertTrue((self.outputs / "report.md").exists())

    def test_main_failure_leaves_submission_untouched(self):
        def fail(prompt, schema):
            raise RuntimeError("Gemini failed after retries: 429")

        code, stderr = self.run_main(["--no-validate"], FakeLLMClient(fail))

        self.assertEqual(code, 1)
        self.assertIn("429", stderr)
        self.assertEqual(self.submission.read_text(encoding="utf-8"), "OLD")

    def test_single_shot_writes_outputs_file(self):
        client = FakeLLMClient(lambda prompt, schema: "Nome: * *")

        code, _ = self.run_main(["--single-shot"], client)

        self.assertEqual(code, 0)
        self.assertEqual(client.prompts[0],
                         config.PROMPT_PATH.read_text(encoding="utf-8"))
        single_shot = self.outputs / "single_shot.txt"
        self.assertEqual(single_shot.read_text(encoding="utf-8"),
                         "Nome: * *")
        self.assertEqual(self.submission.read_text(encoding="utf-8"), "OLD")

    def test_replies_are_cached_unless_no_cache(self):
        for args, refresh in ((["--single-shot"], False),
                              (["--single-shot", "--no-cache"], True)):
            with self.subTest(args=args), \
                    patch.object(config, "main_model", return_value="m"), \
                    patch.object(run, "make_client") as make_client:
                make_client.return_value = FakeLLMClient(
                    lambda prompt, schema: "Nome: * *"
                )

                code, _ = self.run_main(args, None)

                self.assertEqual(code, 0)
                make_client.assert_called_once_with("m", refresh=refresh)


if __name__ == "__main__":
    unittest.main()
