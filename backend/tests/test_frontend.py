"""Frontend tests: the Streamlit app must fail gracefully when the backend is
down. Uses a guaranteed-closed local port; no real backend needed."""
import socket
from pathlib import Path

import pytest
import requests

import api_client
from api_client import BackendError

APP_PATH = Path(__file__).resolve().parents[2] / "frontend" / "app.py"


def _closed_port_url():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}"  # socket closed -> nothing listening


def test_client_uses_backend_url_env(monkeypatch):
    monkeypatch.setenv("BACKEND_URL", "http://example.internal:9000/")
    assert api_client.backend_url() == "http://example.internal:9000"


def test_backend_down_raises_friendly_error(monkeypatch):
    monkeypatch.setenv("BACKEND_URL", _closed_port_url())
    with pytest.raises(BackendError) as exc:
        api_client.get_health()
    assert "Cannot reach the backend" in str(exc.value)
    assert "Traceback" not in str(exc.value)


def test_timeout_raises_friendly_error(monkeypatch):
    def slow(*a, **kw):
        raise requests.exceptions.ReadTimeout()

    monkeypatch.setattr(api_client.requests, "request", slow)
    with pytest.raises(BackendError, match="did not respond"):
        api_client.suggest("VPN down", "VPN keeps disconnecting", "Network", "Low")


def test_validation_error_is_readable(monkeypatch):
    resp = requests.Response()
    resp.status_code = 422
    resp._content = (b'{"detail":[{"loc":["body","title"],'
                     b'"msg":"String should have at least 3 characters"}]}')
    monkeypatch.setattr(api_client.requests, "request", lambda *a, **k: resp)
    with pytest.raises(BackendError, match="title: String should have at least 3"):
        api_client.suggest("x", "VPN keeps disconnecting", "Network", "Low")


def test_streamlit_app_shows_error_when_backend_down(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("BACKEND_URL", _closed_port_url())
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.run()
    assert not at.exception, "app crashed instead of showing an error"
    errors = " ".join(e.value for e in at.error)
    assert "Cannot reach the backend" in errors


def test_streamlit_app_shows_error_on_submit_when_backend_down(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("BACKEND_URL", _closed_port_url())
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.run()
    at.text_input[0].input("VPN keeps disconnecting")
    at.text_area[0].input("My VPN drops every few minutes at home.")
    at.button[0].click()
    at.run()
    assert not at.exception
    assert sum("Cannot reach the backend" in e.value for e in at.error) >= 2


def test_backend_url_defaults_to_localhost(monkeypatch):
    monkeypatch.delenv("BACKEND_URL", raising=False)
    assert api_client.backend_url() == "http://127.0.0.1:8000"
