import os
from pathlib import Path

from dotenv import load_dotenv

SUBMISSION_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = SUBMISSION_DIR.parent

PROMPT_PATH = SUBMISSION_DIR / "prompt.txt"
SUBMISSION_PATH = SUBMISSION_DIR / "submission.txt"
OUTPUTS_DIR = SUBMISSION_DIR / "outputs"
# Gitignored (.cache): stored replies are local, per prompt and model.
CACHE_DIR = OUTPUTS_DIR / ".cache"
PDF_PATH = REPO_DIR / "raw_data" / "document_to_anonymize.pdf"

DEFAULT_MODEL = "gemini-3.8-flash"
TEMPERATURE = 0.0
RETRY_CODES = {429, 500, 503}
MAX_RETRIES = 3
MAX_RETRY_DELAY = 60.0
RETRY_MARGIN = 1.0
# Server errors (500/503, dropped connections) wait 5, 15, 45, 60, 60 s:
# "high demand" spikes outlast a few seconds of backoff, so they also get
# more retries than a 429, whose wait the server states.
MAX_SERVER_RETRIES = 5
BACKOFF_BASE = 5.0
BACKOFF_FACTOR = 3

# The lowest free-tier limit seen (gemini-3.6-flash). Google counts
# requests over a sliding minute, so requests exactly 60 / RPM seconds
# apart can still put RPM + 1 inside one window; the margin prevents it.
REQUESTS_PER_MINUTE = 5
MIN_REQUEST_INTERVAL = 60 / REQUESTS_PER_MINUTE + RETRY_MARGIN

# Every setting loads .env itself, so callers may read a model name before
# anything else touched the environment. load_dotenv never overrides
# variables already set in the shell.


def api_key() -> str:
    _load_env()
    key = os.getenv("API_KEY")
    if not key:
        raise RuntimeError("API_KEY environment variable is not set")
    return key


def main_model() -> str:
    _load_env()
    return os.getenv("MAIN_MODEL", DEFAULT_MODEL)


def data_gen_model() -> str:
    _load_env()
    return os.getenv("DATA_GEN_MODEL") or main_model()


def validation_model() -> str | None:
    """The judge model; None means the meaning score is skipped."""
    _load_env()
    return os.getenv("VALIDATION_MODEL")


def _load_env() -> None:
    load_dotenv(REPO_DIR / ".env")
