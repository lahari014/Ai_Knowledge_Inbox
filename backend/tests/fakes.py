import math

from app.domain.models import ChunkRecord, SearchResult


class MemoryVectorStore:
    def __init__(self) -> None:
        self.records: list[ChunkRecord] = []
        self.deleted: list[str] = []

    def ensure_ready(self) -> None:
        return None

    def upsert(self, records: list[ChunkRecord]) -> None:
        self.records.extend(records)

    def delete_item(self, item_id: str) -> None:
        self.deleted.append(item_id)
        self.records = [record for record in self.records if record.item_id != item_id]

    def count(self) -> int:
        return len(self.records)

    def query(self, embedding: list[float], limit: int) -> list[SearchResult]:
        ranked = sorted(self.records, key=lambda record: _distance(embedding, record.embedding))
        return [
            SearchResult(
                item_id=record.item_id,
                index=record.index,
                text=record.text,
                title=record.title,
                url=record.url,
                source_type=record.source_type,
                distance=_distance(embedding, record.embedding),
            )
            for record in ranked[:limit]
        ]


class FailingVectorStore(MemoryVectorStore):
    def upsert(self, records: list[ChunkRecord]) -> None:
        del records
        from app.api.errors import AppError

        raise AppError("INDEX_UNAVAILABLE", "Vector index is unavailable.", 503)


def _distance(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left)) or 1.0
    right_norm = math.sqrt(sum(value * value for value in right)) or 1.0
    cosine = dot / (left_norm * right_norm)
    return 1.0 - cosine
