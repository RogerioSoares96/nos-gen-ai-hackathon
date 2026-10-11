import unittest

from submission.anonymizer.masking import apply_findings
from submission.anonymizer.schemas import Finding


def F(text, context=""):
    return Finding(text=text, category="x", reason="r", context=context)


class TestApplyFindings(unittest.TestCase):
    def test_multi_word_name(self):
        r = apply_findings("Nome: Maria Conceição Oliveira Santos",
                           [F("Maria Conceição Oliveira Santos")])
        self.assertEqual(r.masked, "Nome: * * * *")

    def test_edge_punctuation_kept(self):
        text = "Morada: Rua das Flores, 123, Sacavém"
        r = apply_findings(text, [F("Flores"), F("Sacavém")])
        self.assertEqual(r.masked, "Morada: Rua das *, 123, *")

    def test_parenthesis_and_comma(self):
        r = apply_findings("Filhos: 2 (João, 15 anos", [F("João")])
        self.assertEqual(r.masked, "Filhos: 2 (*, 15 anos")

    def test_inner_punctuation_is_one_word(self):
        text = "Email: maria.santos@emailpessoal.pt\nCC: 12345678-9ZX0"
        r = apply_findings(text, [F("maria.santos@emailpessoal.pt"),
                                  F("12345678-9ZX0")])
        self.assertEqual(r.masked, "Email: *\nCC: *")

    def test_phone_with_spaces(self):
        r = apply_findings("Telefone: +351 912 345 678",
                           [F("+351 912 345 678")])
        self.assertEqual(r.masked, "Telefone: * * * *")

    def test_every_occurrence_without_context(self):
        r = apply_findings("Maria Santos\nA paciente Maria Santos,",
                           [F("Maria Santos")])
        self.assertEqual(r.masked, "* *\nA paciente * *,")

    def test_context_limits_masking(self):
        text = ("Relatório - Centro Médico Lisboa\n"
                "Morada: Rua das Flores, Sacavém, Lisboa")
        finding = F("Lisboa",
                    context="Morada: Rua das Flores, Sacavém, Lisboa")
        r = apply_findings(text, [finding])
        self.assertEqual(r.masked, "Relatório - Centro Médico Lisboa\n"
                                   "Morada: Rua das Flores, Sacavém, *")

    def test_no_substring_match(self):
        r = apply_findings("Ana fez Anamnese", [F("Ana")])
        self.assertEqual(r.masked, "* fez Anamnese")

    def test_unmatched_reported(self):
        finding = F("Maria  Santos")
        r = apply_findings("Maria Santos", [finding])
        self.assertEqual(r.masked, "Maria Santos")
        self.assertEqual(r.unmatched, [finding])

    def test_blank_finding_ignored(self):
        r = apply_findings("Nome: Maria", [F("  ")])
        self.assertEqual(r.masked, "Nome: Maria")
        self.assertEqual(r.unmatched, [])

    def test_masking_masked_text_is_stable(self):
        r = apply_findings("Nome: * Santos", [F("Santos")])
        self.assertEqual(r.masked, "Nome: * *")


if __name__ == "__main__":
    unittest.main()
