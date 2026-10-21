from datetime import datetime, timezone
from uuid import uuid4

from app.api.errors import AppError
from app.core.config import Settings
from app.core.logging import get_logger
from app.db.repository import ItemRepository
from app.domain.models import ChunkRecord, Item, SourceType
from app.domain.protocols import EmbeddingProvider, PageFetcher, VectorStore
from app.services.chunking import Chunker

logger = get_logger("app.ingestion")


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        repository: ItemRepository,
        chunker: Chunker,
        embeddings: EmbeddingProvider,
        vector_store: VectorStore,
        fetcher: PageFetcher,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._chunker = chunker
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._fetcher = fetcher

    def ingest_note(self, text: str, title: str | None) -> Item:
        cleaned = text.strip()
        self._validate_content(cleaned)
        item = self._build_item(
            source_type=SourceType.NOTE,
            title=(title or "").strip() or derive_title(cleaned, "Untitled note"),
            url=None,
            raw_content=cleaned,
        )
        return self._persist_and_index(item)

    def ingest_url(self, url: str) -> Item:
        fetched = self._fetcher.fetch(url.strip())
        self._validate_content(fetched.text)
        item = self._build_item(
            source_type=SourceType.URL,
            title=fetched.title.strip() or derive_title(fetched.text, "Untitled page"),
            url=fetched.url,
            raw_content=fetched.text,
        )
        return self._persist_and_index(item)

    def _persist_and_index(self, item: Item) -> Item:
        self._repository.add(item)
        logger.info(
            "item_persisted", item_id=item.id, source_type=item.source_type.value, stage="persisted"
        )
        try:
            self._index(item)
        except AppError:
            logger.info("item_index_failed", item_id=item.id, stage="compensate")
            self._compensate(item.id)
            raise
        except Exception as exc:
            logger.info("item_index_failed", item_id=item.id, stage="compensate")
            self._compensate(item.id)
            raise AppError("INDEX_UNAVAILABLE", "The content could not be indexed.", 503) from exc
        logger.info("item_indexed", item_id=item.id, stage="indexed")
        return item

    def _index(self, item: Item) -> None:
        chunks = self._chunker.split(item.raw_content)
        if not chunks:
            raise AppError("EMPTY_CONTENT", "The content did not contain indexable text.", 422)
        embeddings = self._embeddings.embed([chunk.text for chunk in chunks])
        records = [
            ChunkRecord(
                item_id=item.id,
                index=chunk.index,
                text=chunk.text,
                embedding=embedding,
                source_type=item.source_type.value,
                title=item.title,
                url=item.url,
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]
        self._vector_store.upsert(records)

    def _compensate(self, item_id: str) -> None:
        try:
            self._vector_store.delete_item(item_id)
        finally:
            self._repository.delete(item_id)

    def _validate_content(self, content: str) -> None:
        if not content:
            raise AppError("VALIDATION_ERROR", "Content cannot be empty.", 422)
        if len(content) > self._settings.content_max_chars:
            raise AppError("VALIDATION_ERROR", "Content exceeds the configured length limit.", 422)

    def _build_item(
        self, source_type: SourceType, title: str, url: str | None, raw_content: str
    ) -> Item:
        return Item(
            id=str(uuid4()),
            source_type=source_type,
            title=title[:300],
            url=url,
            raw_content=raw_content,
            created_at=datetime.now(timezone.utc),
        )


def derive_title(text: str, fallback: str) -> str:
    normalized = " ".join(text.split())
    if not normalized:
        return fallback
    return normalized[:80]
