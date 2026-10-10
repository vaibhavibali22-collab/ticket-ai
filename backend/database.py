"""Database connection: one engine, a session factory, and helpers."""
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from config import get_settings


class Base(DeclarativeBase):
    """Parent class for all ORM models (tables)."""


def make_engine(url: str):
    # SQLite + FastAPI: requests run on different threads, so allow that.
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    # pre_ping re-checks pooled connections (Neon closes idle ones).
    return create_engine(url, connect_args=connect_args, pool_pre_ping=True)


engine = make_engine(get_settings().database_url)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    """Create tables if they don't exist. Safe to call many times."""
    import models  # noqa: F401  (registers the Ticket table on Base)

    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency: one session per request, always closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
