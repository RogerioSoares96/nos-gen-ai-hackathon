import os
import unittest
from unittest.mock import Mock, patch

import httpx

from submission.anonymizer import config, llm
from submission.anonymizer.llm import GeminiClient
from submission.anonymizer.schemas import EntailmentScore

SPACING = config.MIN_REQUEST_INTERVAL


class Err(Exception):
    def __init__(self, code, message=""):
        super().__init__(f"HTTP {code} {message}".strip())
        self.code = code


def ok_response(text='{"score": 87}'):
    return Mock(text=text)


class TestGeminiClient(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"API_KEY": "test-key"})
        env.start()
        self.addCleanup(env.stop)

        client_cls = patch("submission.anonymizer.llm.genai.Client")
        self.client_cls = client_cls.start()
        self.addCleanup(client_cls.stop)
        self.generate = self.client_cls.return_value.models.generate_content

        # The request history is shared module state; each test starts
        # with none, so no test waits on another's requests.
        history = patch.dict(llm._last_request, clear=True)
        history.start()
        self.addCleanup(history.stop)

        # A fake clock that only moves when the client sleeps, so tests
        # never really wait and every sleep is recorded.
        self.now = 0.0
        self.sleep = Mock(side_effect=self._advance)
        self.attempt_times = []

    def _advance(self, seconds):
        self.now += seconds

    def _clock(self):
        return self.now

    def make(self, model="gemini-x"):
        return GeminiClient(model, sleep=self.sleep, clock=self._clock)

    def record_attempts(self, outcomes):
        """Make generate_content yield outcomes, logging each call time."""
        remaining = list(outcomes)

        def attempt(**kwargs):
            self.attempt_times.append(self.now)
            outcome = remaining.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        self.generate.side_effect = attempt

    def test_generate_json_sends_schema_and_parses(self):
        self.generate.return_value = ok_response()

        result = self.make().generate_json("p", EntailmentScore)

        self.assertEqual(result.score, 87)
        self.client_cls.assert_called_once_with(api_key="test-key")
        kwargs = self.generate.call_args.kwargs
        self.assertEqual(kwargs["model"], "gemini-x")
        self.assertEqual(kwargs["contents"], "p")
        self.assertEqual(kwargs["config"].temperature, 0.0)
        self.assertEqual(
            kwargs["config"].response_mime_type, "application/json"
        )
        self.assertIs(kwargs["config"].response_schema, EntailmentScore)

    def test_generate_text_returns_plain_text(self):
        self.generate.return_value = ok_response("Nome: * *")

        result = self.make().generate_text("p")

        self.assertEqual(result, "Nome: * *")
        config_arg = self.generate.call_args.kwargs["config"]
        self.assertIsNone(config_arg.response_mime_type)

    def test_spaces_consecutive_requests(self):
        self.record_attempts([ok_response(), ok_response()])
        client = self.make()

        client.generate_text("p")
        client.generate_text("p")

        self.assertEqual(self.attempt_times, [0, SPACING])

    def test_no_wait_once_spacing_has_passed(self):
        self.generate.return_value = ok_response()
        client = self.make()

        client.generate_text("p")
        self.now += SPACING + 30
        client.generate_text("p")

        self.sleep.assert_not_called()

    def test_clients_of_one_model_share_spacing(self):
        self.record_attempts([ok_response(), ok_response()])

        self.make().generate_text("p")
        self.make().generate_text("p")

        self.assertEqual(self.attempt_times, [0, SPACING])

    def test_models_are_spaced_independently(self):
        self.record_attempts([ok_response(), ok_response()])

        self.make("model-a").generate_text("p")
        self.make("model-b").generate_text("p")

        self.assertEqual(self.attempt_times, [0, 0])
        self.sleep.assert_not_called()

    def test_retries_on_429_then_succeeds(self):
        self.generate.side_effect = [Err(429), Err(503), ok_response()]

        result = self.make().generate_json("p", EntailmentScore)

        self.assertEqual(result.score, 87)
        self.assertEqual(self.generate.call_count, 3)

    def test_server_errors_back_off_in_growing_steps(self):
        self.record_attempts([Err(503), Err(503), Err(503), ok_response()])

        self.make().generate_json("p", EntailmentScore)

        # The first 5 s backoff is topped up to the request spacing.
        self.assertEqual(
            [c.args[0] for c in self.sleep.call_args_list],
            [5, SPACING - 5, 15, 45],
        )
        self.assertEqual(
            self.attempt_times, [0, SPACING, SPACING + 15, SPACING + 60]
        )

    def test_server_error_backoff_is_capped(self):
        self.record_attempts([Err(503)] * 5 + [ok_response()])

        self.make().generate_json("p", EntailmentScore)

        self.assertEqual(
            [c.args[0] for c in self.sleep.call_args_list],
            [5, SPACING - 5, 15, 45,
             config.MAX_RETRY_DELAY, config.MAX_RETRY_DELAY],
        )

    def test_server_errors_get_more_retries_than_rate_limits(self):
        self.generate.side_effect = Err(503)

        with self.assertRaises(RuntimeError) as ctx:
            self.make().generate_json("p", EntailmentScore)

        self.assertIn("503", str(ctx.exception))
        self.assertEqual(
            self.generate.call_count, config.MAX_SERVER_RETRIES + 1
        )
        self.assertGreater(config.MAX_SERVER_RETRIES, config.MAX_RETRIES)

    def test_gives_up_after_max_retries(self):
        self.generate.side_effect = Err(429)

        with self.assertRaises(RuntimeError) as ctx:
            self.make().generate_json("p", EntailmentScore)

        self.assertIn("429", str(ctx.exception))
        self.assertEqual(self.generate.call_count, config.MAX_RETRIES + 1)

    def test_retries_dropped_connection(self):
        reset = ConnectionResetError(10054, "connection forcibly closed")
        self.generate.side_effect = [reset, ok_response()]

        result = self.make().generate_json("p", EntailmentScore)

        self.assertEqual(result.score, 87)
        self.assertEqual(self.generate.call_count, 2)

    def test_retries_http_transport_error(self):
        self.generate.side_effect = [
            httpx.RemoteProtocolError("server disconnected"), ok_response()
        ]

        result = self.make().generate_json("p", EntailmentScore)

        self.assertEqual(result.score, 87)

    def test_429_waits_exact_delay_from_message_plus_margin(self):
        # RetryInfo rounds the delay down ('20s'); the message keeps the
        # fraction, and waiting the rounded value retries too early.
        self.generate.side_effect = [
            Err(429, "Please retry in 20.657348117s. "
                     "{'retryDelay': '20s'}"),
            ok_response(),
        ]

        self.make().generate_json("p", EntailmentScore)

        self.sleep.assert_called_once()
        self.assertAlmostEqual(
            self.sleep.call_args.args[0], 20.657348117 + config.RETRY_MARGIN
        )

    def test_429_falls_back_to_retry_info(self):
        self.generate.side_effect = [
            Err(429, "PerMinute ... {'retryDelay': '32s'}"), ok_response()
        ]

        self.make().generate_json("p", EntailmentScore)

        self.sleep.assert_called_once_with(32.0 + config.RETRY_MARGIN)

    def test_server_retry_delay_is_capped(self):
        self.generate.side_effect = [
            Err(429, "{'retryDelay': '600s'}"), ok_response()
        ]

        self.make().generate_json("p", EntailmentScore)

        self.sleep.assert_called_once_with(config.MAX_RETRY_DELAY)

    def test_daily_quota_fails_fast(self):
        self.generate.side_effect = Err(
            429, "quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier"
        )

        with self.assertRaises(RuntimeError) as ctx:
            self.make().generate_json("p", EntailmentScore)

        self.assertIn("daily quota", str(ctx.exception))
        self.assertEqual(self.generate.call_count, 1)
        self.sleep.assert_not_called()

    def test_non_retryable_error_raises_immediately(self):
        self.generate.side_effect = Err(400)

        with self.assertRaises(Err):
            self.make().generate_json("p", EntailmentScore)

        self.assertEqual(self.generate.call_count, 1)
        self.sleep.assert_not_called()


class TestConfig(unittest.TestCase):
    def setUp(self):
        # Never read the developer's real .env from a test.
        dotenv = patch("submission.anonymizer.config.load_dotenv")
        self.load_dotenv = dotenv.start()
        self.addCleanup(dotenv.stop)

    def env(self, **values):
        return patch.dict(os.environ, values, clear=True)

    def test_missing_api_key_raises(self):
        with self.env():
            with self.assertRaises(RuntimeError):
                config.api_key()

    def test_main_model_default(self):
        with self.env():
            self.assertEqual(config.main_model(), config.DEFAULT_MODEL)

    def test_main_model_reads_env(self):
        with self.env(MAIN_MODEL="models/gemini-2.5-flash"):
            self.assertEqual(config.main_model(), "models/gemini-2.5-flash")

    def test_model_lookup_loads_dotenv_first(self):
        with self.env():
            config.main_model()
        self.load_dotenv.assert_called_with(config.REPO_DIR / ".env")

    def test_data_gen_model_falls_back_to_main(self):
        with self.env(MAIN_MODEL="main"):
            self.assertEqual(config.data_gen_model(), "main")
        with self.env(MAIN_MODEL="main", DATA_GEN_MODEL="gen"):
            self.assertEqual(config.data_gen_model(), "gen")

    def test_validation_model_is_optional(self):
        with self.env():
            self.assertIsNone(config.validation_model())
        with self.env(VALIDATION_MODEL="judge"):
            self.assertEqual(config.validation_model(), "judge")


if __name__ == "__main__":
    unittest.main()
