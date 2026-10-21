from collections.abc import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.models import Base, ItemRecord
from app.db.session import make_engine, make_session_factory
from app.domain.models import Item, ItemSummary, SourceType


class ItemRepository:
    def __init__(
        self, settings: Settings, session_factory: Callable[[], Session] | None = None
    ) -> None:
        self._settings = settings
        self._engine = make_engine(settings)
        self._session_factory = session_factory or make_session_factory(self._engine)

    def create_schema(self) -> None:
        Base.metadata.create_all(self._engine)

    def add(self, item: Item) -> Item:
        with self._session_factory() as session:
            session.add(
                ItemRecord(
                    id=item.id,
                    source_type=item.source_type.value,
                    title=item.title,
                    url=item.url,
                    raw_content=item.raw_content,
                    created_at=item.created_at,
                )
            )
            session.commit()
        return item

    def delete(self, item_id: str) -> None:
        with self._session_factory() as session:
            record = session.get(ItemRecord, item_id)
            if record is not None:
                session.delete(record)
                session.commit()

    def list_newest(self) -> list[ItemSummary]:
        with self._session_factory() as session:
            records = session.scalars(
                select(ItemRecord).order_by(ItemRecord.created_at.desc(), ItemRecord.id.desc())
            ).all()
        return [self.to_summary(record) for record in records]

    def to_summary(self, record: ItemRecord) -> ItemSummary:
        return ItemSummary(
            id=record.id,
            source_type=SourceType(record.source_type),
            title=record.title,
            url=record.url,
            preview=build_preview(record.raw_content, self._settings.preview_chars),
            created_at=record.created_at,
        )


def build_preview(raw_content: str, limit: int) -> str:
    normalized = " ".join(raw_content.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit]}…"
