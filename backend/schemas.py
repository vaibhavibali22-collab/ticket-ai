"""Pydantic schemas: the shape of every request and response.
FastAPI uses them to validate input (bad input -> HTTP 422) and to build /docs."""
from datetime import datetime
from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class Category(str, Enum):
    network = "Network"
    email = "Email"
    hardware = "Hardware"
    software = "Software"
    access = "Access"
    other = "Other"


class Priority(str, Enum):
    low = "Low"
    medium = "Medium"
    high = "High"
    critical = "Critical"


class Status(str, Enum):
    open = "Open"
    resolved = "Resolved"


class _Strict(BaseModel):
    # Strip spaces, so "   " counts as empty and fails min_length.
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


class TicketInput(_Strict):
    title: str = Field(..., min_length=3, max_length=200, examples=["VPN keeps disconnecting"])
    description: str = Field(
        ..., min_length=10, max_length=5000,
        examples=["My VPN drops every few minutes when I work from home."],
    )
    category: Category = Category.other
    priority: Priority = Priority.medium


class TicketCreate(TicketInput):
    # If a resolution is supplied the ticket is stored as Resolved and
    # immediately becomes searchable history for future suggestions.
    resolution: Optional[str] = Field(None, min_length=10, max_length=5000)


class TicketOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    description: str
    category: str
    priority: str
    status: str
    resolution: Optional[str]
    created_at: datetime


class TicketPage(BaseModel):
    items: List[TicketOut]
    total: int
    limit: int
    offset: int


class SuggestRequest(TicketInput):
    save_ticket: bool = Field(False, description="Also store this ticket as Open.")


class Match(BaseModel):
    id: int
    title: str
    description: str
    category: str
    priority: str
    resolution: str
    similarity: float = Field(
        ..., ge=0, le=1,
        description="Cosine similarity of TF-IDF vectors (0-1). NOT a probability.",
    )


class SubmittedTicket(BaseModel):
    id: Optional[int] = Field(None, description="Set only when save_ticket=true.")
    title: str
    description: str
    category: str
    priority: str


class SuggestResponse(BaseModel):
    ticket: SubmittedTicket
    matches: List[Match]
    resolution_steps: List[str] = Field(..., min_length=5, max_length=5)
    generation_mode: Literal["llm", "fallback"]
    model: Optional[str] = Field(None, description="LLM model id, only when mode is llm.")
    similarity_threshold: float
    warnings: List[str]


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    database: Literal["ok", "error"]
    database_backend: str
    historical_tickets: int = Field(..., description="All tickets (Open + Resolved).")
    resolved_tickets: int = Field(..., description="Resolved tickets used for matching.")
    llm_configured: bool
    llm_model: str
    similarity_threshold: float
