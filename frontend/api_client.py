"""Thin HTTP client for the FastAPI backend. The frontend never runs NLP or
LLM code itself; it only calls these functions. Every failure is turned into a
BackendError with a short, user-friendly message (no tracebacks)."""
import os
from pathlib import Path
from typing import Any, Dict, Optional

import requests

try:  # optional: read BACKEND_URL from backend/.env if present
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / "backend" / ".env", override=False)
except ImportError:
    pass

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"
# Generous timeouts: a free cloud backend that went to sleep can take ~30-60 s
# to wake up. "Connection refused" (backend not running) still fails instantly.
TIMEOUT_SHORT = 30    # seconds: health and listing calls
TIMEOUT_SUGGEST = 90  # seconds: retrieval + LLM generation


class BackendError(Exception):
    """Raised with a message that is safe to show in the UI."""


def _from_streamlit_secrets() -> Optional[str]:
    """On Streamlit Community Cloud, settings live in the app's 'Secrets'."""
    try:
        import streamlit as st

        value = st.secrets.get("BACKEND_URL")
        return str(value) if value else None
    except Exception:  # no secrets file locally -> just ignore
        return None


def backend_url() -> str:
    """Order: BACKEND_URL env var -> Streamlit secrets -> local default."""
    url = os.getenv("BACKEND_URL") or _from_streamlit_secrets() or DEFAULT_BACKEND_URL
    return url.rstrip("/")


def _validation_message(resp: requests.Response) -> str:
    try:
        detail = resp.json().get("detail")
    except ValueError:
        return "The request was rejected by the backend."
    if isinstance(detail, list):  # FastAPI 422 format
        parts = []
        for err in detail:
            field = ".".join(str(p) for p in err.get("loc", [])[1:]) or "input"
            parts.append(f"{field}: {err.get('msg', 'invalid value')}")
        return "Please fix: " + "; ".join(parts)
    return str(detail or "The request was rejected by the backend.")


def _request(method: str, path: str, timeout: float, **kwargs) -> Any:
    url = f"{backend_url()}{path}"
    try:
        resp = requests.request(method, url, timeout=timeout, **kwargs)
    except requests.exceptions.Timeout:
        raise BackendError(f"The backend did not respond within {timeout} seconds. "
                           "Please try again.") from None
    except requests.exceptions.ConnectionError:
        raise BackendError(f"Cannot reach the backend at {backend_url()}. Start it with "
                           "'uvicorn main:app --reload' in the backend folder.") from None
    except requests.exceptions.RequestException:
        raise BackendError("Unexpected error while contacting the backend.") from None

    if resp.status_code == 422:
        raise BackendError(_validation_message(resp))
    if resp.status_code == 404:
        raise BackendError("Not found.")
    if resp.status_code >= 400:
        raise BackendError(f"The backend returned an error (HTTP {resp.status_code}).")
    try:
        return resp.json()
    except ValueError:
        raise BackendError("The backend returned an unreadable response.") from None


def get_health() -> Dict:
    return _request("GET", "/health", TIMEOUT_SHORT)


def suggest(title: str, description: str, category: str, priority: str,
            save_ticket: bool = False) -> Dict:
    payload = {"title": title, "description": description, "category": category,
               "priority": priority, "save_ticket": save_ticket}
    return _request("POST", "/tickets/suggest", TIMEOUT_SUGGEST, json=payload)


def list_tickets(limit: int = 20, offset: int = 0, status: Optional[str] = None) -> Dict:
    params = {"limit": limit, "offset": offset}
    if status:
        params["status"] = status
    return _request("GET", "/tickets", TIMEOUT_SHORT, params=params)
