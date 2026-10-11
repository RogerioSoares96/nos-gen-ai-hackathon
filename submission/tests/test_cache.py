import json
import tempfile
import unittest
from pathlib import Path

from submission.anonymizer.llm import CachedClient
from submission.anonymizer.schemas import EntailmentScore
from submission.tests.fakes import FakeLLMClient


def score_reply(prompt, schema):
    return json.dumps({"score": 87})


class TestCachedClient(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name) / ".cache"

    def make(self, responder=score_reply, model="gemini-x", refresh=False):
        inner = FakeLLMClient(responder)
        cached = CachedClient(inner, model, self.directory, refresh=refresh)
        return cached, inner

    def test_json_reply_is_reused(self):
        cached, inner = self.make()

        first = cached.generate_json("p", EntailmentScore)
        second = cached.generate_json("p", EntailmentScore)

        self.assertEqual((first.score, second.score), (87, 87))
        self.assertEqual(len(inner.prompts), 1)

    def test_cache_survives_a_new_client(self):
        self.make()[0].generate_json("p", EntailmentScore)

        cached, inner = self.make()
        result = cached.generate_json("p", EntailmentScore)

        self.assertEqual(result.score, 87)
        self.assertEqual(inner.prompts, [])

    def test_text_reply_is_reused(self):
        cached, inner = self.make(lambda prompt, schema: "Nome: * *")

        cached.generate_text("p")
        result = cached.generate_text("p")

        self.assertEqual(result, "Nome: * *")
        self.assertEqual(len(inner.prompts), 1)

    def test_other_prompt_calls_again(self):
        cached, inner = self.make()

        cached.generate_json("p1", EntailmentScore)
        cached.generate_json("p2", EntailmentScore)

        self.assertEqual(inner.prompts, ["p1", "p2"])

    def test_other_model_calls_again(self):
        self.make(model="model-a")[0].generate_json("p", EntailmentScore)

        cached, inner = self.make(model="model-b")
        cached.generate_json("p", EntailmentScore)

        self.assertEqual(inner.prompts, ["p"])

    def test_failure_is_not_cached(self):
        replies = [RuntimeError("Gemini failed after retries: 503"),
                   json.dumps({"score": 87})]

        def flaky(prompt, schema):
            reply = replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply

        cached, inner = self.make(flaky)
        with self.assertRaises(RuntimeError):
            cached.generate_json("p", EntailmentScore)
        result = cached.generate_json("p", EntailmentScore)

        self.assertEqual(result.score, 87)
        self.assertEqual(len(inner.prompts), 2)

    def test_empty_text_reply_is_not_cached(self):
        cached, inner = self.make(lambda prompt, schema: None)

        cached.generate_text("p")
        cached.generate_text("p")

        self.assertEqual(len(inner.prompts), 2)

    def test_refresh_asks_again_and_replaces_stored_reply(self):
        self.make()[0].generate_json("p", EntailmentScore)

        refreshing, inner = self.make(
            lambda prompt, schema: json.dumps({"score": 42}), refresh=True
        )
        result = refreshing.generate_json("p", EntailmentScore)

        self.assertEqual(result.score, 42)
        self.assertEqual(len(inner.prompts), 1)
        reused = self.make()[0].generate_json("p", EntailmentScore)
        self.assertEqual(reused.score, 42)


if __name__ == "__main__":
    unittest.main()
