import json
from types import SimpleNamespace

import llm_service

VPN = {"title": "VPN keeps dropping", "category": "Network", "priority": "High",
       "description": "My VPN disconnects every few minutes when working from home."}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["database"] == "ok"
    assert body["database_backend"] == "sqlite"
    assert body["llm_configured"] is False
    assert "gsk_" not in r.text


def test_create_ticket(client):
    r = client.post("/tickets", json={"title": "Printer jam", "category": "Hardware",
                                      "description": "Paper keeps jamming in tray 2."})
    assert r.status_code == 201
    body = r.json()
    assert body["id"] > 0 and body["status"] == "Open" and body["priority"] == "Medium"
    assert client.get(f"/tickets/{body['id']}").json()["title"] == "Printer jam"


def test_create_ticket_with_resolution_is_resolved(client):
    r = client.post("/tickets", json={**VPN, "resolution": "Switched the VPN protocol to TCP."})
    assert r.status_code == 201 and r.json()["status"] == "Resolved"


def test_create_ticket_validation_errors(client):
    bad_payloads = [
        {},                                                    # missing fields
        {"title": "   ", "description": "long enough description"},  # blank title
        {"title": "Valid title", "description": "short"},      # too short
        {"title": "Valid title", "description": "long enough description",
         "priority": "Whenever"},                              # bad enum
        {"title": "Valid title", "description": "long enough description",
         "unknown": 1},                                        # unexpected field
    ]
    for payload in bad_payloads:
        assert client.post("/tickets", json=payload).status_code == 422, payload


def test_get_missing_ticket_404(client):
    assert client.get("/tickets/99999").status_code == 404


def test_list_tickets_pagination(client, seeded):
    page1 = client.get("/tickets", params={"limit": 10}).json()
    page2 = client.get("/tickets", params={"limit": 10, "offset": 10}).json()
    assert page1["total"] >= 50 and len(page1["items"]) == 10
    assert {t["id"] for t in page1["items"]}.isdisjoint({t["id"] for t in page2["items"]})
    assert client.get("/tickets", params={"status": "Open"}).json()["total"] == 0


def test_list_tickets_rejects_bad_limit(client):
    assert client.get("/tickets", params={"limit": 1000}).status_code == 422
    assert client.get("/tickets", params={"offset": -1}).status_code == 422
    assert client.get("/tickets", params={"status": "Pending"}).status_code == 422


def test_suggest_returns_full_response(client, seeded):
    r = client.post("/tickets/suggest", json=VPN)
    assert r.status_code == 200
    body = r.json()
    assert body["ticket"]["title"] == VPN["title"] and body["ticket"]["id"] is None
    assert body["matches"][0]["title"] == "VPN disconnects frequently"
    scores = [m["similarity"] for m in body["matches"]]
    assert scores == sorted(scores, reverse=True)
    assert all(m["resolution"] for m in body["matches"])
    assert len(body["resolution_steps"]) == 5
    assert body["generation_mode"] == "fallback"  # no key in tests
    assert body["model"] is None
    assert any("Fallback mode" in w for w in body["warnings"])
    assert body["similarity_threshold"] == 0.15


def test_suggest_rejects_blank_title(client):
    r = client.post("/tickets/suggest", json={**VPN, "title": "  "})
    assert r.status_code == 422


def test_suggest_with_empty_database(client):
    r = client.post("/tickets/suggest", json=VPN)
    assert r.status_code == 200
    body = r.json()
    assert body["matches"] == []
    assert len(body["resolution_steps"]) == 5
    assert any("database is empty" in w for w in body["warnings"])


def test_suggest_no_relevant_match_warns(client, seeded):
    r = client.post("/tickets/suggest", json={
        "title": "Coffee machine", "description": "The pantry coffee machine is leaking water."})
    body = r.json()
    assert body["matches"] == [] and len(body["resolution_steps"]) == 5
    assert any("No sufficiently similar historical ticket" in w for w in body["warnings"])


def test_suggest_save_ticket_creates_open_ticket(client, seeded):
    body = client.post("/tickets/suggest", json={**VPN, "save_ticket": True}).json()
    new_id = body["ticket"]["id"]
    assert new_id is not None
    assert client.get(f"/tickets/{new_id}").json()["status"] == "Open"


def test_suggest_with_mocked_llm(client, seeded, monkeypatch, fake_key):
    steps = [f"Mocked LLM step {i} with clear instructions." for i in range(1, 6)]
    response = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=json.dumps({"steps": steps})))])
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: response)))
    monkeypatch.setattr(llm_service, "_create_client", lambda s: fake)

    body = client.post("/tickets/suggest", json=VPN).json()
    assert body["generation_mode"] == "llm"
    assert body["model"] == "llama-3.1-8b-instant"
    assert body["resolution_steps"] == steps
    assert not any("Fallback" in w for w in body["warnings"])


def test_suggest_groq_failure_still_returns_200(client, seeded, monkeypatch, fake_key):
    def boom(**kw):
        import groq, httpx
        raise groq.APIConnectionError(request=httpx.Request("POST", "https://x"))

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=boom)))
    monkeypatch.setattr(llm_service, "_create_client", lambda s: fake)
    r = client.post("/tickets/suggest", json=VPN)
    assert r.status_code == 200
    assert r.json()["generation_mode"] == "fallback"
    assert len(r.json()["resolution_steps"]) == 5


def test_health_counts_resolved_and_open(client, seeded):
    client.post("/tickets", json={"title": "Printer jam", "description": "Paper keeps jamming in tray 2."})
    body = client.get("/health").json()
    assert body["resolved_tickets"] == body["historical_tickets"] - 1


def test_auto_seed_on_startup(monkeypatch):
    from fastapi.testclient import TestClient
    from main import app
    from seed_db import SEED_TICKETS

    monkeypatch.setenv("AUTO_SEED", "true")
    with TestClient(app) as c:          # startup runs the seed
        pass
    with TestClient(app) as c:          # second boot must not duplicate
        assert c.get("/health").json()["resolved_tickets"] == len(SEED_TICKETS)


def test_no_auto_seed_by_default(client):
    assert client.get("/health").json()["historical_tickets"] == 0
