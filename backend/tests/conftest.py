"""Shared test setup.

Environment variables are set BEFORE any app module is imported, so tests
always use a throwaway SQLite file and never a real Groq key or network.
"""
import os
import sys
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="ticket_ai_tests_"))
os.environ["DATABASE_URL"] = f"sqlite:///{(_TMP / 'test.db').as_posix()}"
os.environ["GROQ_API_KEY"] = ""          # tests never use a real key
os.environ["GROQ_MODEL"] = "llama-3.1-8b-instant"
os.environ["SIMILARITY_THRESHOLD"] = "0.15"
os.environ["LLM_MAX_ATTEMPTS"] = "2"

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR.parent / "frontend"))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from database import Base, SessionLocal, engine, init_db  # noqa: E402
import models  # noqa: E402,F401


@pytest.fixture(autouse=True)
def fresh_database():
    """Every test starts with empty tables."""
    Base.metadata.drop_all(bind=engine)
    init_db()
    yield


@pytest.fixture
def db():
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
def seeded(db):
    from seed_db import seed

    seed(db)
    return db


@pytest.fixture
def client():
    from main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def fake_key(monkeypatch):
    """Pretend a Groq key is configured (calls are still mocked)."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_dummy_key_123")
    return "gsk_test_dummy_key_123"
