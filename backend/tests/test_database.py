from sqlalchemy import inspect

import config
import crud
from database import engine
from seed_db import SEED_TICKETS, seed


def test_init_db_creates_tickets_table():
    inspector = inspect(engine)
    assert "tickets" in inspector.get_table_names()
    columns = {c["name"] for c in inspector.get_columns("tickets")}
    assert {"id", "title", "description", "category", "priority", "status",
            "resolution", "created_at"} <= columns


def test_seed_inserts_50_tickets(db):
    inserted, skipped = seed(db)
    assert inserted == len(SEED_TICKETS) >= 50
    assert skipped == 0
    assert crud.count_tickets(db) == len(SEED_TICKETS)
    # every seeded ticket is resolved, so it is usable history
    assert len(crud.get_resolved_tickets(db)) == len(SEED_TICKETS)


def test_seed_is_idempotent(db):
    seed(db)
    inserted, skipped = seed(db)
    assert inserted == 0
    assert skipped == len(SEED_TICKETS)
    assert crud.count_tickets(db) == len(SEED_TICKETS)


def test_seed_restores_only_missing_rows(db):
    seed(db)
    first = crud.get_ticket(db, 1)
    db.delete(first)
    db.commit()
    inserted, _ = seed(db)
    assert inserted == 1
    assert crud.count_tickets(db) == len(SEED_TICKETS)


def test_seed_covers_required_topics():
    text = " ".join(f"{t[2]} {t[3]}" for t in SEED_TICKETS).lower()
    for topic in ["wi-fi", "internet", "vpn", "password", "locked", "sync", "printer",
                  "slow", "install", "crash", "dns", "permission", "access denied",
                  "disk space", "network drive", "microphone", "sound", "browser",
                  "mfa", "log in"]:
        assert topic in text, f"seed data missing topic: {topic}"


def test_postgres_url_is_normalised():
    assert config.normalise_database_url("postgres://u:p@h/db") == "postgresql://u:p@h/db"
    assert config.normalise_database_url("sqlite:///x.db") == "sqlite:///x.db"


def test_default_sqlite_path_is_inside_backend(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    url = config.get_settings().database_url
    assert url.startswith("sqlite:///")
    assert url.endswith("/backend/tickets.db")  # absolute, independent of cwd
