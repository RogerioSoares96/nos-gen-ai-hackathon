import unittest

from submission.anonymizer import config
from submission.anonymizer.extract import clean_lines, pdf_to_text


class TestExtract(unittest.TestCase):
    def test_clean_lines_drops_empty_and_trailing_space(self):
        raw = "Relatório\n\n  \nA paciente relatou dores \nabdominais\n"
        self.assertEqual(
            clean_lines(raw),
            "Relatório\nA paciente relatou dores\nabdominais",
        )

    def test_real_pdf_header(self):
        text = pdf_to_text(config.PDF_PATH)
        self.assertEqual(text.splitlines()[:4], [
            "Relatório de Admissão - Centro Médico Lisboa",
            "Data: 15 de abril de 2025",
            "Referência: ADM-2025-04-15-089",
            "Informações do Paciente:",
        ])
        self.assertIn("Nome: Maria Conceição Oliveira Santos", text)


if __name__ == "__main__":
    unittest.main()
