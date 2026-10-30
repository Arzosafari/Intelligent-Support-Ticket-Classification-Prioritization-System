"""Strict API schemas; whitespace-only text is not a valid ticket."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from support_ticket.features.text import normalize_text


class Ticket(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ticket_id: Annotated[StrictStr, Field(min_length=1, max_length=128)] | None = None
    text: Annotated[StrictStr, Field(min_length=1, max_length=10000)]

    @field_validator("text")
    @classmethod
    def not_blank(cls, value: str) -> str:
        normalize_text(value)
        return value


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tickets: list[Ticket] = Field(min_length=1, max_length=100)


class TaskPrediction(BaseModel):
    label: str
    score: float | None
    score_type: Literal["uncalibrated_probability", "unavailable"]
    review_required: bool
    review_reason: str


class Prediction(BaseModel):
    ticket_id: str | None = None
    model_version: str
    category: TaskPrediction
    priority: TaskPrediction | None
    priority_status: Literal["available", "unavailable"]


class BatchResponse(BaseModel):
    predictions: list[Prediction]
