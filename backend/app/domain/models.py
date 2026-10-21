from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class SourceType(str, Enum):
    NOTE = "note"
    URL = "url"


@dataclass(frozen=True)
class Item:
    id: str
    source_type: SourceType
    title: str
    url: str | None
    raw_content: str
    created_at: datetime


@dataclass(frozen=True)
class ItemSummary:
    id: str
    source_type: SourceType
    title: str
    url: str | None
    preview: str
    created_at: datetime


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str


@dataclass(frozen=True)
class ChunkRecord:
    item_id: str
    index: int
    text: str
    embedding: list[float]
    source_type: str
    title: str
    url: str | None


@dataclass(frozen=True)
class SearchResult:
    item_id: str
    index: int
    text: str
    title: str
    url: str | None
    source_type: str
    distance: float


@dataclass(frozen=True)
class Context:
    number: int
    label: str
    text: str


@dataclass(frozen=True)
class Citation:
    number: int
    item_id: str
    label: str
    url: str | None
    snippet: str
    score: float


@dataclass(frozen=True)
class QueryResult:
    answer: str
    mode: str
    citations: list[Citation]


@dataclass(frozen=True)
class FetchedContent:
    url: str
    title: str
    text: str
