"""All database reads/writes live here (CRUD = Create, Read, Update, Delete),
so routes and the NLP code never write SQL themselves."""
from typing import List, Optional, Tuple

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from models import Ticket


def create_ticket(db: Session, *, title: str, description: str, category: str,
                  priority: str, resolution: Optional[str] = None) -> Ticket:
    ticket = Ticket(
        title=title, description=description, category=category, priority=priority,
        resolution=resolution, status="Resolved" if resolution else "Open",
    )
    db.add(ticket)
    db.commit()
    db.refresh(ticket)
    return ticket


def get_ticket(db: Session, ticket_id: int) -> Optional[Ticket]:
    return db.get(Ticket, ticket_id)


def list_tickets(db: Session, *, limit: int = 20, offset: int = 0,
                 status: Optional[str] = None,
                 category: Optional[str] = None) -> Tuple[List[Ticket], int]:
    query = select(Ticket)
    if status:
        query = query.where(Ticket.status == status)
    if category:
        query = query.where(Ticket.category == category)
    total = db.scalar(select(func.count()).select_from(query.subquery())) or 0
    items = db.scalars(query.order_by(Ticket.id.desc()).limit(limit).offset(offset)).all()
    return list(items), total


def get_resolved_tickets(db: Session) -> List[dict]:
    """Historical knowledge base for the NLP engine: resolved tickets only."""
    rows = db.scalars(
        select(Ticket)
        .where(Ticket.status == "Resolved", Ticket.resolution.is_not(None))
        .order_by(Ticket.id)
    ).all()
    return [t.to_dict() for t in rows if (t.resolution or "").strip()]


def count_tickets(db: Session, status: Optional[str] = None) -> int:
    query = select(func.count(Ticket.id))
    if status:
        query = query.where(Ticket.status == status)
    return db.scalar(query) or 0


def ticket_exists(db: Session, title: str, description: str) -> bool:
    return db.scalar(
        select(Ticket.id).where(Ticket.title == title, Ticket.description == description)
    ) is not None
