from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.domain.models import SourceType


def to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.capitalize() for part in tail)


class ApiModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class NoteIngest(BaseModel):
    type: Literal["note"]
    text: str
    title: str | None = None

    @field_validator("text")
    @classmethod
    def text_is_present(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Note text cannot be empty")
        return value


class UrlIngest(BaseModel):
    type: Literal["url"]
    url: str

    @field_validator("url")
    @classmethod
    def url_is_present(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("URL cannot be empty")
        return value.strip()


IngestRequest = Annotated[NoteIngest | UrlIngest, Field(discriminator="type")]


class QueryRequest(BaseModel):
    question: str
    limit: int | None = None


class ItemSummarySchema(ApiModel):
    id: str
    source_type: SourceType
    title: str
    url: str | None = None
    preview: str
    created_at: datetime


class ItemResponse(ApiModel):
    item: ItemSummarySchema


class ItemListResponse(ApiModel):
    items: list[ItemSummarySchema]


class CitationSchema(ApiModel):
    number: int
    item_id: str
    label: str
    url: str | None = None
    snippet: str
    score: float


class QueryResponse(ApiModel):
    answer: str
    mode: Literal["demo", "openai"]
    citations: list[CitationSchema]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    mode: Literal["demo", "openai"]
