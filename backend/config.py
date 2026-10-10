"""Central configuration. Every setting comes from environment variables
(optionally loaded from backend/.env) with a safe default, so the app runs
locally with zero configuration."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent

# Load backend/.env no matter which directory the app is started from.
# override=False: real environment variables win over the .env file.
load_dotenv(BACKEND_DIR / ".env", override=False)

DEFAULT_SQLITE_URL = f"sqlite:///{(BACKEND_DIR / 'tickets.db').as_posix()}"

# Groq's Llama 3.1 8B model id (as named in the problem statement).
# Groq announced this model's retirement for free/developer tiers, so if your
# key gets a "model not found / decommissioned" error, set
# GROQ_MODEL=openai/gpt-oss-20b (Groq's recommended replacement).
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def normalise_database_url(url: str) -> str:
    """Neon/Heroku often give 'postgres://...', SQLAlchemy needs 'postgresql://'."""
    if url.startswith("postgres://"):
        return "postgresql://" + url[len("postgres://"):]
    return url


@dataclass(frozen=True)
class Settings:
    database_url: str
    groq_api_key: str
    groq_model: str
    groq_timeout_seconds: float
    llm_max_attempts: int
    similarity_threshold: float
    top_k: int
    auto_seed: bool


def get_settings() -> Settings:
    """Read settings at call time (cheap) so tests can change env vars."""
    return Settings(
        database_url=normalise_database_url(os.getenv("DATABASE_URL") or DEFAULT_SQLITE_URL),
        groq_api_key=(os.getenv("GROQ_API_KEY") or "").strip(),
        groq_model=(os.getenv("GROQ_MODEL") or DEFAULT_GROQ_MODEL).strip(),
        groq_timeout_seconds=_float("GROQ_TIMEOUT_SECONDS", 20.0),
        # Number of times we ask the LLM again if its output fails validation.
        llm_max_attempts=max(1, min(_int("LLM_MAX_ATTEMPTS", 2), 5)),
        # Initial value chosen by inspecting scores on the synthetic data:
        # relevant matches scored ~0.25-0.80, unrelated ones mostly < 0.12.
        # MUST be re-validated on real ticket data before production use.
        similarity_threshold=_float("SIMILARITY_THRESHOLD", 0.15),
        top_k=max(1, min(_int("TOP_K", 5), 10)),
        auto_seed=(os.getenv("AUTO_SEED") or "").strip().lower() in ("1", "true", "yes"),
    )
