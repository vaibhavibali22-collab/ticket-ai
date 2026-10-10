# IT Ticket Resolution Suggestion Engine
*AI + NLP-based support assistant: HCLTech AI Engineer hackathon MVP*

## 1. Problem statement
IT service desks receive the same kinds of issues again and again (VPN drops, lockouts, printers, Outlook). Agents and employees lose time searching old tickets for fixes. We want a system that, given a new ticket, **instantly suggests a solution** based on how similar tickets were resolved before.

## 2. Proposed solution
A hybrid retrieval + generation pipeline:

1. **Retrieve.** TF-IDF + cosine similarity (scikit-learn) ranks *resolved* historical tickets by textual similarity to the new ticket. Only matches above a similarity threshold are kept.
2. **Generate.** The new ticket and the matched tickets with their resolutions are sent to an LLM on Groq (Llama 3.1 8B by default). The LLM writes **exactly five** actionable steps.
3. **Validate.** Our code checks for exactly 5 distinct, non-empty, safe steps, retries a bounded number of times, and otherwise uses a **clearly labelled deterministic fallback**.

The app works fully **without** a Groq key, running in fallback mode, so the demo never depends on the network.

## 3. Architecture

```mermaid
flowchart LR
    U[User] --> FE[Streamlit UI<br/>frontend/app.py]
    FE -- HTTP JSON --> API[FastAPI<br/>backend/main.py]
    API --> V[Pydantic validation<br/>schemas.py]
    API --> DB[(SQLite / PostgreSQL<br/>SQLAlchemy: models.py, crud.py)]
    DB -- resolved tickets --> NLP[TF-IDF + cosine similarity<br/>nlp_engine.py]
    NLP -- top matches + scores --> LLM[llm_service.py]
    LLM -- prompt --> G[Groq API<br/>Llama 3.1 8B]
    G -- JSON steps --> CHK{Exactly 5 valid,<br/>safe steps?}
    CHK -- yes --> RES[Response: mode = llm]
    CHK -- "no (retry, bounded)" --> G
    LLM -- no key / API error / still invalid --> FB[Deterministic fallback]
    FB --> RES2[Response: mode = fallback]
    RES --> FE
    RES2 --> FE
```

## 4. Technology stack and why

| Component | Why |
|---|---|
| **Streamlit** | Builds a clean dashboard in pure Python, which is fast to build and easy to demo |
| **FastAPI + Uvicorn** | Typed, fast REST API with automatic validation and Swagger docs at `/docs` |
| **SQLAlchemy + SQLite** | Zero-setup local database. The same code works with PostgreSQL/Neon via `DATABASE_URL` |
| **scikit-learn TF-IDF + cosine similarity** | Fast, explainable and deterministic retrieval with no GPU, training or model download |
| **Groq (Llama 3.1 8B)** | Very low-latency hosted LLM to turn raw past fixes into clear steps |
| **Pydantic** | Request/response validation; bad input returns HTTP 422 automatically |
| **pytest** | 116 automated tests; Groq is mocked, so no key or internet is needed |

## 5. End-to-end request flow
1. The user fills in title, description, category and priority, then clicks **Suggest resolutions**.
2. Streamlit sends `POST /tickets/suggest` to FastAPI (`frontend/api_client.py`).
3. FastAPI validates the body with Pydantic. A blank or too-short title or description returns **422**.
4. `crud.get_resolved_tickets` loads resolved tickets from the database.
5. `nlp_engine.find_similar_tickets` fits TF-IDF on those tickets, transforms the new ticket with the same vectorizer, computes cosine similarity, sorts (ties broken by ticket id), and keeps the top 5 above the threshold.
6. `llm_service.generate_resolution` either:
   - calls Groq with JSON output and validates exactly 5 steps, retrying up to `LLM_MAX_ATTEMPTS`, or
   - uses the deterministic fallback when there is no key, an API error, or still-invalid output.
7. The response contains the ticket, matches with similarity scores and resolutions, the 5 steps, `generation_mode` (`llm`/`fallback`) and warnings.
8. Streamlit shows the steps, a green **LLM** or amber **Fallback** badge, warnings and the matches.

## 6. Project structure
```
ticket-ai/
├── README.md               ← this file
├── TRACEABILITY.md         ← requirement → file → test matrix
├── TEACHING_GUIDE.md       ← how everything works + judge Q&A
├── .gitignore
├── backend/
│   ├── main.py             ← FastAPI routes
│   ├── config.py           ← all settings from environment variables
│   ├── database.py         ← SQLAlchemy engine + session
│   ├── models.py           ← Ticket table
│   ├── schemas.py          ← Pydantic request/response models
│   ├── crud.py             ← database reads/writes
│   ├── nlp_engine.py       ← TF-IDF + cosine similarity retrieval
│   ├── llm_service.py      ← Groq call, validation, safety filter, fallback
│   ├── seed_db.py          ← 52 synthetic resolved tickets (idempotent)
│   ├── requirements.txt
│   ├── .env.example
│   └── tests/              ← pytest suite (116 tests)
└── frontend/
    ├── app.py              ← Streamlit dashboard
    └── api_client.py       ← HTTP calls to the backend + friendly errors
```

## 7–13. Setup and run

You need Python **3.10 or newer**. The tests were run on 3.10 and 3.13.

### macOS / Linux
```bash
cd ticket-ai/backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # optional: add GROQ_API_KEY
python seed_db.py             # safe to run more than once
uvicorn main:app --reload     # backend on http://127.0.0.1:8000
```
Open a **second terminal**:
```bash
cd ticket-ai/backend && source venv/bin/activate
cd ../frontend
streamlit run app.py          # opens http://localhost:8501
```

### Windows (PowerShell)
```powershell
cd ticket-ai\backend
python -m venv venv
venv\Scripts\Activate.ps1      # cmd.exe: venv\Scripts\activate.bat
pip install -r requirements.txt
copy .env.example .env         # optional: add GROQ_API_KEY
python seed_db.py
uvicorn main:app --reload
```
Second terminal:
```powershell
cd ticket-ai\backend; venv\Scripts\Activate.ps1
cd ..\frontend
streamlit run app.py
```
If PowerShell blocks the activate script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

### Environment configuration (`backend/.env`)
| Variable | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` | *(empty)* | Groq key. Leave it empty to use fallback mode |
| `GROQ_MODEL` | `llama-3.1-8b-instant` in code; `openai/gpt-oss-20b` in `.env.example` | Groq model id (see the note below) |
| `DATABASE_URL` | SQLite file `backend/tickets.db` | e.g. Neon: `postgresql://user:pass@host/db?sslmode=require` |
| `BACKEND_URL` | `http://127.0.0.1:8000` | Where Streamlit finds the API |
| `SIMILARITY_THRESHOLD` | `0.15` | Minimum cosine similarity for a match |
| `TOP_K` | `5` | Maximum matches returned |
| `GROQ_TIMEOUT_SECONDS` | `20` | Timeout for each Groq request |
| `LLM_MAX_ATTEMPTS` | `2` | Attempts if the LLM output fails validation |
| `AUTO_SEED` | `false` | `true` loads the seed tickets at startup (for cloud hosts) |

> **Model note.** Groq's deprecations page announced that `llama-3.1-8b-instant` would be shut down on 16 Aug 2026 for free/developer tiers. With a free key it now fails with "model unavailable", so `.env.example` uses **`openai/gpt-oss-20b`**, Groq's recommended replacement, which has been verified working with a real key. If the app shows *"model … is unavailable"*, set `GROQ_MODEL=openai/gpt-oss-20b`, Groq's recommended replacement, in `.env` and restart the backend. Either way, the app never crashes; it falls back.

## 14. API examples
Interactive docs are at **http://127.0.0.1:8000/docs**.

```bash
curl http://127.0.0.1:8000/health

curl "http://127.0.0.1:8000/tickets?limit=5&offset=0&status=Resolved"

curl -X POST http://127.0.0.1:8000/tickets -H "Content-Type: application/json" \
  -d '{"title":"Printer jam","description":"Paper keeps jamming in tray 2","category":"Hardware","priority":"Low"}'

curl -X POST http://127.0.0.1:8000/tickets/suggest -H "Content-Type: application/json" \
  -d '{"title":"VPN keeps dropping","description":"My VPN disconnects every few minutes when I work from home","category":"Network","priority":"High"}'
```
Example `/tickets/suggest` response, captured from a real run in fallback mode and shortened here:
```json
{
  "ticket": {"id": null, "title": "VPN keeps dropping", "category": "Network", "priority": "High", "...": "..."},
  "matches": [
    {"id": 6, "title": "VPN disconnects frequently", "similarity": 0.6326, "resolution": "Use a wired connection ..."},
    {"id": 1, "title": "Wi-Fi keeps disconnecting", "similarity": 0.2083, "resolution": "Forget the Wi-Fi network ..."}
  ],
  "resolution_steps": [
    "Use a wired connection or move closer to the router. (from similar ticket #6)",
    "Update the VPN client to the latest approved version. (from similar ticket #6)",
    "Switch the VPN protocol to TCP in the client settings. (from similar ticket #6)",
    "Check whether other websites or apps work, and whether colleagues nearby have the same problem.",
    "Because this ticket is marked High, contact the IT service desk now with the error messages and the steps you have already tried."
  ],
  "generation_mode": "fallback",
  "model": null,
  "similarity_threshold": 0.15,
  "warnings": ["Fallback mode: the LLM was not used (GROQ_API_KEY is not set). ..."]
}
```
On Windows PowerShell, use `curl.exe` (not `curl`) or the Swagger page at `/docs`.

## 15. Testing
```bash
cd backend
python -m pytest -v
```
The tests use a temporary SQLite file and a mocked Groq client. They need **no API key and no internet**. Coverage includes DB init and seeding (including re-seeding), TF-IDF ranking, threshold, ties, duplicates, empty input and empty DB, exact-five validation, the safety filter, retries, every Groq error type, the fallback, every endpoint, HTTP 404/422, and Streamlit behaviour when the backend is down (using `streamlit.testing`).

## 16. Known limitations
- **TF-IDF is lexical.** It matches words, not meaning: "can't sign in" vs "authentication failure" may not match. Sentence embeddings would fix this; we kept TF-IDF because the problem statement specifies it and it is explainable.
- **The similarity threshold (0.15) and the 2-shared-word guard were tuned on synthetic data only.** Re-validate them on real tickets.
- **The TF-IDF model is refit on every request.** That's fine for hundreds of tickets; at scale you would cache it and refit only when tickets change.
- **The safety filter is a keyword/regex list.** It reduces risk but is not a complete guarantee.
- **The live Groq path depends on your key, model availability and rate limits.** Automated tests mock it.
- **There is no authentication or user accounts**, which is out of scope for the MVP. The seed tickets are synthetic.

## 17. Troubleshooting
| Symptom | Fix |
|---|---|
| UI says *"Cannot reach the backend"* | Start `uvicorn main:app --reload` in `backend/`; check `BACKEND_URL` |
| Sidebar says the database is empty | Run `python seed_db.py` in `backend/` |
| Badge says *Fallback* with "GROQ_API_KEY is not set" | Add the key to `backend/.env` and restart uvicorn |
| Warning says *"model … is unavailable"* | Set `GROQ_MODEL=openai/gpt-oss-20b` in `.env` and restart |
| Warning says *rate limit* | Wait a minute; the fallback keeps working meanwhile |
| `ModuleNotFoundError` | Activate the venv, then `pip install -r requirements.txt` |
| Port 8000 already in use | `uvicorn main:app --port 8001` and set `BACKEND_URL=http://127.0.0.1:8001` |
| Neon connection fails | The URL must start with `postgresql://` (or `postgres://`) and include `?sslmode=require` |

## 18. Deploying online (free)
The backend goes on **Render** (free web service, configured by `render.yaml`). The frontend goes on **Streamlit Community Cloud** (free).
1. Push the project to a GitHub repository. `.gitignore` keeps `.env`, `venv/` and `*.db` out.
2. Go to render.com and choose **New → Blueprint**, then select the repo. Render reads `render.yaml`. Enter your `GROQ_API_KEY` when asked. When the deploy finishes, open `https://<your-service>.onrender.com/health` to confirm it is running.
3. Go to share.streamlit.io and choose **Create app**, then pick the repo with main file path `frontend/app.py`. Under **Advanced settings**, choose Python 3.11 and add this to Secrets: `BACKEND_URL = "https://<your-service>.onrender.com"`.
4. Free Render services sleep after a period of inactivity. The first request then takes up to about a minute, so open the `/health` URL a minute before you present.
5. The free disk is temporary. `AUTO_SEED=true` reloads the 52 seed tickets on every start, but tickets you add are lost when it restarts. For permanent storage, set `DATABASE_URL` to a Neon Postgres URL.

The full click-by-click guide is in the teaching PDF.

## 19. Hackathon demo workflow (about 5 minutes)
1. **Show `/docs`.** Point out the endpoints and the Pydantic schemas.
2. **Strong match.** Enter "VPN keeps dropping / disconnects every few minutes from home". Show the top match (#6, about 0.63) and its resolution, then the 5 steps and the mode badge.
3. **No match.** Enter "Pantry coffee machine is leaking". Show that there are no matches, the warning, and the 5 cautious general steps ending in escalation.
4. **Validation.** Submit an empty title and show the friendly error.
5. **Resilience.** Remove or invalidate `GROQ_API_KEY`, restart, re-run, and show the *Fallback* badge. Nothing crashes.
6. **Quality.** Run `python -m pytest -q` and show 116 passing tests.
7. Close with the architecture diagram, the limitations and next steps.
