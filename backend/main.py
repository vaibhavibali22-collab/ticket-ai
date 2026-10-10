"""FastAPI app: HTTP routes only. Database work is in crud.py, retrieval in
nlp_engine.py, generation in llm_service.py.

Routes are plain `def` (not `async def`) on purpose: they do blocking work
(database, scikit-learn, Groq HTTP call), and FastAPI runs `def` routes in a
thread pool, so one slow request does not freeze the server's event loop.
"""
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

import crud
from config import get_settings
from database import engine, get_db, init_db
from llm_service import generate_resolution
from nlp_engine import find_similar_tickets, ticket_text
from schemas import (Category, HealthOut, Match, Status, SubmittedTicket, SuggestRequest,
                     SuggestResponse, TicketCreate, TicketOut, TicketPage)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ticket_ai.api")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()  # create tables on startup if missing
    settings = get_settings()
    # Cloud hosts with a temporary disk (e.g. Render free tier) start with an
    # empty SQLite file, so AUTO_SEED=true loads the 52 seed tickets on boot.
    # seed() is idempotent, so this never creates duplicates.
    if settings.auto_seed:
        from seed_db import seed

        inserted, _ = seed()
        logger.info("AUTO_SEED: inserted %d seed tickets", inserted)
    logger.info("Started. DB=%s, LLM configured=%s, model=%s, threshold=%.2f",
                engine.dialect.name, bool(settings.groq_api_key), settings.groq_model,
                settings.similarity_threshold)
    yield


app = FastAPI(
    title="IT Ticket Resolution Suggestion Engine",
    description="TF-IDF + cosine similarity retrieval over historical tickets, "
                "then an LLM (Groq) writes exactly five resolution steps.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(SQLAlchemyError)
def database_error_handler(_request: Request, exc: SQLAlchemyError):
    logger.error("Database error: %s", type(exc).__name__)
    return JSONResponse(status_code=503, content={"detail": "Database unavailable. Try again shortly."})


@app.get("/health", response_model=HealthOut, tags=["system"])
def health(db: Session = Depends(get_db)):
    settings = get_settings()
    try:
        count, resolved = crud.count_tickets(db), crud.count_tickets(db, "Resolved")
        db_state = "ok"
    except SQLAlchemyError:
        count, resolved, db_state = 0, 0, "error"
    return HealthOut(
        status="ok" if db_state == "ok" else "degraded",
        database=db_state,
        database_backend=engine.dialect.name,
        historical_tickets=count,
        resolved_tickets=resolved,
        llm_configured=bool(settings.groq_api_key),  # never the key itself
        llm_model=settings.groq_model,
        similarity_threshold=settings.similarity_threshold,
    )


@app.get("/tickets", response_model=TicketPage, tags=["tickets"])
def list_tickets(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status_filter: Optional[Status] = Query(None, alias="status"),
    category: Optional[Category] = None,
    db: Session = Depends(get_db),
):
    items, total = crud.list_tickets(
        db, limit=limit, offset=offset,
        status=status_filter.value if status_filter else None,
        category=category.value if category else None)
    return TicketPage(items=items, total=total, limit=limit, offset=offset)


@app.get("/tickets/{ticket_id}", response_model=TicketOut, tags=["tickets"])
def get_ticket(ticket_id: int, db: Session = Depends(get_db)):
    ticket = crud.get_ticket(db, ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket


@app.post("/tickets", response_model=TicketOut, status_code=status.HTTP_201_CREATED,
          tags=["tickets"])
def create_ticket(body: TicketCreate, db: Session = Depends(get_db)):
    ticket = crud.create_ticket(db, title=body.title, description=body.description,
                                category=body.category.value, priority=body.priority.value,
                                resolution=body.resolution)
    logger.info("Created ticket id=%d status=%s", ticket.id, ticket.status)
    return ticket


@app.post("/tickets/suggest", response_model=SuggestResponse, tags=["suggestions"])
def suggest(body: SuggestRequest, db: Session = Depends(get_db)):
    settings = get_settings()
    ticket = {"title": body.title, "description": body.description,
              "category": body.category.value, "priority": body.priority.value}

    # 1. Retrieve: rank resolved historical tickets by TF-IDF cosine similarity.
    history = crud.get_resolved_tickets(db)
    matches = find_similar_tickets(
        ticket_text(body.title, body.description), history,
        top_k=settings.top_k, threshold=settings.similarity_threshold)

    warnings = []
    if not history:
        warnings.append("The historical ticket database is empty, so no evidence was "
                        "available. Run 'python seed_db.py'.")
    elif not matches:
        warnings.append(
            f"No sufficiently similar historical ticket was found (similarity threshold "
            f"{settings.similarity_threshold:.2f}). Steps are general troubleshooting, "
            "not based on past resolutions.")

    # 2. Generate: LLM (or labelled fallback) -> exactly 5 steps.
    result = generate_resolution(ticket, matches, settings)
    warnings.extend(result.warnings)

    # 3. Optionally store the new ticket as Open.
    ticket_id = None
    if body.save_ticket:
        ticket_id = crud.create_ticket(db, **ticket).id

    logger.info("Suggest: category=%s matches=%d mode=%s saved_id=%s",
                ticket["category"], len(matches), result.mode, ticket_id)
    return SuggestResponse(
        ticket=SubmittedTicket(id=ticket_id, **ticket),
        matches=[Match(**{k: m[k] for k in Match.model_fields}) for m in matches],
        resolution_steps=result.steps,
        generation_mode=result.mode,
        model=result.model,
        similarity_threshold=settings.similarity_threshold,
        warnings=warnings,
    )
