import json
import tempfile
import unittest
from pathlib import Path

from submission.anonymizer.orchestrator import PipelineResult
from submission.anonymizer.report import write_report
from submission.anonymizer.schemas import Finding


def make_result():
    return PipelineResult(
        masked="Nome: * *",
        findings=[Finding(text="Maria Santos", category="identity",
                          reason="Nome da paciente", context="Nome: ...")],
        rejected=["Lisboa"],
        unmatched=[Finding(text="Mariana", category="identity",
                           reason="r")],
    )


class TestWriteReport(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.out = Path(directory.name) / "outputs"

    def test_report_files_written(self):
        write_report(make_result(), self.out)

        data = json.loads(
            (self.out / "findings.json").read_text(encoding="utf-8")
        )
        self.assertEqual(data["findings"][0]["text"], "Maria Santos")
        self.assertEqual(data["rejected"], ["Lisboa"])
        self.assertEqual(data["unmatched"][0]["text"], "Mariana")

    def test_report_shows_unmatched_and_score(self):
        write_report(make_result(), self.out, entailment=92)

        report = (self.out / "report.md").read_text(encoding="utf-8")
        self.assertIn("| Maria Santos | identity | Nome da paciente |",
                      report)
        self.assertIn("Mariana", report)
        self.assertIn("Meaning preservation: 92/100", report)

    def test_report_without_score(self):
        write_report(make_result(), self.out)

        report = (self.out / "report.md").read_text(encoding="utf-8")
        self.assertIn("Meaning preservation: not computed", report)


if __name__ == "__main__":
    unittest.main()
