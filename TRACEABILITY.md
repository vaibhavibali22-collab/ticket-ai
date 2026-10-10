# Requirement Traceability Matrix

Each requirement maps to the file that implements it and the automated test that verifies it.
Test files are in `backend/tests/`. Run them with `python -m pytest -v` from `backend/`.

| # | Requirement | Implemented in | Verified by |
|---|---|---|---|
| R1 | SQLAlchemy `Ticket` model: id, title, description, category, priority, status, resolution, created_at | `models.py` | `test_database.py::test_init_db_creates_tickets_table` |
| R2 | SQLite by default, `DATABASE_URL` override (Neon/Postgres) | `config.py`, `database.py` | `test_database.py::test_postgres_url_is_normalised` |
| R3 | DB path works from any working directory | `config.py` (absolute path) | `test_database.py::test_default_sqlite_path_is_inside_backend` |
| R4 | Idempotent seed with ~50 realistic tickets covering all listed topics | `seed_db.py` | `test_database.py::test_seed_inserts_50_tickets`, `::test_seed_is_idempotent`, `::test_seed_covers_required_topics` |
| R5 | TF-IDF fitted on historical corpus (title + description) | `nlp_engine.py::find_similar_tickets` | `test_nlp_engine.py::test_vpn_query_ranks_vpn_ticket_first` |
| R6 | New ticket transformed with the *same* fitted vectorizer; cosine similarity | `nlp_engine.py` | `test_nlp_engine.py::test_exact_text_scores_highest` |
| R7 | Ranked by descending similarity, top 5 or fewer | `nlp_engine.py` | `test_nlp_engine.py::test_results_sorted_descending_and_limited` |
| R8 | Configurable similarity threshold; irrelevant tickets excluded | `config.py` (`SIMILARITY_THRESHOLD`), `nlp_engine.py` | `test_nlp_engine.py::test_threshold_filters_irrelevant`, `::test_threshold_is_configurable` |
| R9 | Empty input handled | `schemas.py` (422), `nlp_engine.py` (`ValueError`) | `test_nlp_engine.py::test_empty_query_raises`, `test_api.py::test_suggest_rejects_blank_title` |
| R10 | Empty historical database handled | `nlp_engine.py`, `main.py` | `test_nlp_engine.py::test_empty_history_returns_no_matches`, `test_api.py::test_suggest_with_empty_database` |
| R11 | Ties ordered consistently; duplicate tickets collapsed | `nlp_engine.py` | `test_nlp_engine.py::test_ties_broken_by_ticket_id`, `::test_duplicates_collapsed` |
| R12 | Groq SDK; `GROQ_API_KEY` and `GROQ_MODEL` from env; no hardcoded key | `config.py`, `llm_service.py` | `test_llm_service.py::test_missing_api_key_uses_fallback` |
| R13 | Exactly 5 distinct, non-empty steps; validated in code, not just the prompt | `llm_service.py::validate_steps` | `test_llm_service.py::test_validate_*` (8 tests) |
| R14 | Structured output where supported, bounded retries, then fallback | `llm_service.py::_call_llm` | `test_llm_service.py::test_invalid_then_valid_output_retries`, `::test_always_invalid_output_falls_back` |
| R15 | Handles auth failure, rate limit, network error, timeout, model unavailable, empty output | `llm_service.py` | `test_llm_service.py::test_groq_errors_fall_back` (parametrised), `::test_empty_model_output_falls_back` |
| R16 | Ticket text treated as untrusted data (prompt-injection defence) | `llm_service.py::build_messages` | `test_llm_service.py::test_user_text_never_in_system_prompt` |
| R17 | No destructive / security-bypassing steps | `llm_service.py::is_unsafe` | `test_llm_service.py::test_validate_rejects_unsafe_step`, `::test_negated_warning_is_not_unsafe` |
| R18 | Deterministic, labelled fallback with exactly 5 meaningful steps (no filler padding) | `llm_service.py::build_fallback_steps` | `test_llm_service.py::test_fallback_always_five_valid_steps` (every category, with/without matches), `::test_fallback_is_deterministic` |
| R19 | Generation mode reported honestly (`llm` vs `fallback`) | `llm_service.py`, `schemas.py` | `test_llm_service.py::test_successful_llm_call_reports_llm_mode`, `test_api.py::test_suggest_with_mocked_llm` |
| R20 | No secrets or raw errors leaked | `llm_service.py`, `main.py` | `test_llm_service.py::test_groq_errors_fall_back` (asserts key not in warnings) |
| R21 | `GET /health` with non-sensitive diagnostics | `main.py` | `test_api.py::test_health` |
| R22 | `GET /tickets` with pagination | `main.py`, `crud.py` | `test_api.py::test_list_tickets_pagination`, `::test_list_tickets_rejects_bad_limit` |
| R23 | `POST /tickets` (201) with validation (422) | `main.py`, `schemas.py` | `test_api.py::test_create_ticket`, `::test_create_ticket_validation_errors` |
| R24 | `POST /tickets/suggest` returns ticket, matches, scores, resolutions, 5 steps, mode, warnings | `main.py` | `test_api.py::test_suggest_returns_full_response` |
| R25 | Appropriate HTTP errors (404, 422) | `main.py` | `test_api.py::test_get_missing_ticket_404` |
| R26 | Sync routes so blocking work runs in FastAPI's threadpool | `main.py` (`def`, not `async def`) | Code review (see TEACHING_GUIDE.md) |
| R27 | Streamlit calls FastAPI only; `BACKEND_URL` configurable | `frontend/app.py`, `frontend/api_client.py` | `test_frontend.py::test_client_uses_backend_url_env` |
| R28 | Frontend handles backend down / timeout gracefully | `frontend/api_client.py`, `frontend/app.py` | `test_frontend.py::test_backend_down_raises_friendly_error`, `::test_timeout_raises_friendly_error`, `::test_streamlit_app_shows_error_when_backend_down` |
| R29 | `.env.example`, `.gitignore` | `backend/.env.example`, `.gitignore` | Manual check (final report) |
| R30 | README: Mermaid diagram, mac/Windows setup, API examples, troubleshooting, demo script | `README.md` | Manual check (final report) |
| R31 | Cloud deployment: auto-seed on boot for temporary disks (idempotent) | `main.py` lifespan, `config.py` (`AUTO_SEED`), `render.yaml` | `test_api.py::test_auto_seed_on_startup`, `::test_no_auto_seed_by_default` |
| R32 | Health shows resolved (knowledge base) vs all tickets | `main.py`, `crud.py`, `schemas.py` | `test_api.py::test_health_counts_resolved_and_open` |
| R33 | Frontend reads `BACKEND_URL` from env, Streamlit secrets, or default | `frontend/api_client.py` | `test_frontend.py::test_client_uses_backend_url_env`, `::test_backend_url_defaults_to_localhost` |
