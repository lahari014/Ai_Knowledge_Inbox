from typing import Protocol

from app.domain.models import ChunkRecord, Context, FetchedContent, SearchResult


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class VectorStore(Protocol):
    def ensure_ready(self) -> None: ...

    def upsert(self, records: list[ChunkRecord]) -> None: ...

    def query(self, embedding: list[float], limit: int) -> list[SearchResult]: ...

    def delete_item(self, item_id: str) -> None: ...

    def count(self) -> int: ...


class AnswerProvider(Protocol):
    def answer(self, question: str, contexts: list[Context]) -> str: ...


class PageFetcher(Protocol):
    def fetch(self, url: str) -> FetchedContent: ...
