import json
import unittest

from submission.anonymizer.orchestrator import run_pipeline
from submission.tests.fakes import FakeLLMClient

DETECTOR = "detetor de dados sensíveis"
CONTEXT = "verificação de contexto"
REVIEWER = "revisão final"


def finding(text, category="identity", context=""):
    return {"text": text, "category": category, "reason": "r",
            "context": context}


def responder(detections=None, rejections=None, review=None):
    """Route each prompt to its agent by the template's heading."""

    def respond(prompt, schema):
        if DETECTOR in prompt:
            return json.dumps({"findings": detections or []})
        if CONTEXT in prompt:
            return json.dumps({"additions": [],
                               "rejections": rejections or []})
        if REVIEWER in prompt:
            return json.dumps({"findings": review or []})
        raise AssertionError("unknown agent prompt")

    return respond


def calls_to(client, heading):
    return [p for p in client.prompts if heading in p]


class TestRunPipeline(unittest.TestCase):
    def test_end_to_end_masks_detected_and_reviewed(self):
        client = FakeLLMClient(responder(
            detections=[finding("Maria Santos")],
            review=[finding("Católica", category="sensitive_social")],
        ))

        result = run_pipeline("Nome: Maria Santos\nReligião: Católica",
                              "BASE", client)

        self.assertEqual(result.masked, "Nome: * *\nReligião: *")
        self.assertEqual(
            [f.text for f in result.findings], ["Maria Santos", "Católica"]
        )

    def test_rejections_remove_findings(self):
        client = FakeLLMClient(responder(
            detections=[finding("Maria"), finding("Santos")],
            rejections=["Santos"],
        ))

        result = run_pipeline("Nome: Maria\nApelido: Santos", "BASE", client)

        self.assertEqual(result.masked, "Nome: *\nApelido: Santos")
        self.assertEqual(result.rejected, ["Santos"])

    def test_three_calls_when_something_is_found(self):
        client = FakeLLMClient(responder(detections=[finding("Maria")]))

        run_pipeline("Nome: Maria", "BASE", client)

        self.assertEqual(len(calls_to(client, DETECTOR)), 1)
        self.assertEqual(len(calls_to(client, CONTEXT)), 1)
        self.assertEqual(len(calls_to(client, REVIEWER)), 1)
        self.assertEqual(len(client.prompts), 3)

    def test_context_check_skipped_without_findings(self):
        client = FakeLLMClient(responder())

        result = run_pipeline("Sexo: Feminino", "BASE", client)

        self.assertEqual(calls_to(client, CONTEXT), [])
        self.assertEqual(len(client.prompts), 2)
        self.assertEqual(result.masked, "Sexo: Feminino")

    def test_findings_deduplicated(self):
        same = finding("Maria", context="Nome: Maria")
        client = FakeLLMClient(responder(detections=[same, same]))

        result = run_pipeline("Nome: Maria", "BASE", client)

        self.assertEqual(len(result.findings), 1)

    def test_unmatched_collected(self):
        client = FakeLLMClient(responder(detections=[finding("Mariana")]))

        result = run_pipeline("Nome: Maria", "BASE", client)

        self.assertEqual([f.text for f in result.unmatched], ["Mariana"])


if __name__ == "__main__":
    unittest.main()
