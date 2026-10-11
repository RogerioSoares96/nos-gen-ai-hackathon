import hashlib
import itertools
import re
import time
from pathlib import Path
from typing import Protocol, TypeVar

import httpx
from google import genai
from google.genai import types
from pydantic import BaseModel

from submission.anonymizer import config

T = TypeVar("T", bound=BaseModel)

# When each model was last sent a request (monotonic seconds). Kept at
# module level because the quota is per model, not per client: run.py and
# the validator each build their own GeminiClient.
_last_request: dict[str, float] = {}


class LLMClient(Protocol):
    def generate_json(self, prompt: str, schema: type[T]) -> T: ...

    def generate_text(self, prompt: str) -> str: ...


class GeminiClient:
    """Gemini via google-genai, spacing requests to stay under the
    per-minute quota and retrying rate-limit and server errors."""

    def __init__(self, model: str, temperature: float = config.TEMPERATURE,
                 sleep=time.sleep, clock=time.monotonic):
        self.model = model
        self.temperature = temperature
        self._sleep = sleep
        self._clock = clock
        self._client = genai.Client(api_key=config.api_key())

    def generate_json(self, prompt: str, schema: type[T]) -> T:
        response = self._generate(prompt, types.GenerateContentConfig(
            temperature=self.temperature,
            response_mime_type="application/json",
            response_schema=schema,
        ))
        # Validating ourselves (instead of response.parsed) makes a bad
        # reply raise pydantic's ValidationError rather than return None.
        return schema.model_validate_json(response.text)

    def generate_text(self, prompt: str) -> str:
        response = self._generate(prompt, types.GenerateContentConfig(
            temperature=self.temperature,
        ))
        return response.text

    def _generate(self, prompt: str, generation_config):
        for attempt in itertools.count():
            # Retries wait for a slot too, in case rejected requests also
            # count against the quota.
            self._wait_for_slot()
            try:
                return self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=generation_config,
                )
            except Exception as error:
                if _is_daily_quota(error):
                    # Retrying within seconds can't help: the quota only
                    # resets the next day.
                    raise RuntimeError(
                        f"Gemini daily quota exhausted: {error}"
                    ) from error

                if not _is_retryable(error):
                    raise

                if attempt >= _max_retries(error):
                    raise RuntimeError(
                        f"Gemini failed after retries: {error}"
                    ) from error

                self._sleep(_retry_delay(error, attempt))

    def _wait_for_slot(self) -> None:
        last = _last_request.get(self.model)
        if last is not None:
            remaining = last + config.MIN_REQUEST_INTERVAL - self._clock()
            if remaining > 0:
                self._sleep(remaining)

        _last_request[self.model] = self._clock()


class CachedClient:
    """Stores each successful reply on disk, keyed by model and prompt.

    A failed run then costs only the calls that never succeeded: a rerun
    reads the others back. A changed prompt is a new key, so it is always
    sent; refresh=True ignores stored replies and replaces them.
    """

    def __init__(self, client: LLMClient, model: str, directory: Path,
                 refresh: bool = False):
        self._client = client
        self._model = model
        self._directory = directory
        self._refresh = refresh

    def generate_json(self, prompt: str, schema: type[T]) -> T:
        path = self._path(schema.__name__, prompt)
        stored = self._read(path)
        if stored is not None:
            return schema.model_validate_json(stored)

        result = self._client.generate_json(prompt, schema)
        self._write(path, result.model_dump_json())
        return result

    def generate_text(self, prompt: str) -> str:
        path = self._path("text", prompt)
        stored = self._read(path)
        if stored is not None:
            return stored

        text = self._client.generate_text(prompt)
        # Gemini returns no text for a blocked reply; storing nothing would
        # turn a one-off failure into a permanent one.
        if text is not None:
            self._write(path, text)
        return text

    def _path(self, kind: str, prompt: str) -> Path:
        key = "\0".join((self._model, kind, prompt)).encode("utf-8")
        return self._directory / f"{hashlib.sha256(key).hexdigest()}.txt"

    def _read(self, path: Path) -> str | None:
        if self._refresh or not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def _write(self, path: Path, content: str) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        # newline="\n" keeps a cached reply byte-identical to a fresh one.
        with open(path, "w", encoding="utf-8", newline="\n") as file:
            file.write(content)


def make_client(model: str, refresh: bool = False) -> LLMClient:
    """The client the commands use: Gemini behind the reply cache."""
    return CachedClient(GeminiClient(model), model, config.CACHE_DIR,
                        refresh=refresh)


def _is_retryable(error: Exception) -> bool:
    # Dropped connections (e.g. Windows' WinError 10054) carry no HTTP
    # code, so they are recognised by type rather than by status.
    if isinstance(error, (ConnectionError, TimeoutError,
                          httpx.TransportError)):
        return True
    return getattr(error, "code", None) in config.RETRY_CODES


def _max_retries(error: Exception) -> int:
    if getattr(error, "code", None) == 429:
        return config.MAX_RETRIES
    return config.MAX_SERVER_RETRIES


def _retry_delay(error: Exception, attempt: int) -> float:
    # A per-minute 429 says how long to wait; a fixed backoff would retry
    # too early and fail every time.
    if getattr(error, "code", None) == 429:
        delay = _server_delay(str(error))
        if delay is not None:
            return min(delay + config.RETRY_MARGIN, config.MAX_RETRY_DELAY)
    backoff = config.BACKOFF_BASE * config.BACKOFF_FACTOR ** attempt
    return min(backoff, config.MAX_RETRY_DELAY)


def _server_delay(message: str) -> float | None:
    # The message keeps the exact delay ('retry in 20.657348117s');
    # RetryInfo rounds it down ('retryDelay': '20s'), so it's the fallback.
    for pattern in (r"retry in (\d+(?:\.\d+)?)s",
                    r"'retryDelay': '(\d+(?:\.\d+)?)s'"):
        match = re.search(pattern, message)
        if match:
            return float(match.group(1))
    return None


def _is_daily_quota(error: Exception) -> bool:
    return (getattr(error, "code", None) == 429
            and "PerDay" in str(error))
