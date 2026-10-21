from app.api.errors import AppError
from app.core.config import Settings
from app.domain.models import ChunkRecord, SearchResult


class ChromaVectorStore:
    """Chroma adapter. The rest of the application never sees Chroma's response shape."""

    def __init__(self, settings: Settings, collection: object | None = None) -> None:
        self._settings = settings
        self._collection = collection

    def ensure_ready(self) -> None:
        collection = self._get_collection()
        metadata = getattr(collection, "metadata", None) or {}
        stored = metadata.get("embedding_dimension")
        expected = self._settings.embedding_dimensions
        if stored is not None and int(stored) != expected:
            raise AppError(
                "CONFIGURATION_ERROR",
                "The vector collection dimensions do not match the selected embedding provider. "
                "Create a new collection or switch back to the provider that indexed this data.",
                500,
            )

    def upsert(self, records: list[ChunkRecord]) -> None:
        if not records:
            return
        self._get_collection().upsert(  # type: ignore[attr-defined]
            ids=[f"{record.item_id}:{record.index}" for record in records],
            documents=[record.text for record in records],
            embeddings=[record.embedding for record in records],
            metadatas=[_metadata(record) for record in records],
        )

    def query(self, embedding: list[float], limit: int) -> list[SearchResult]:
        available = self.count()
        if limit < 1 or available == 0:
            return []
        response = self._get_collection().query(  # type: ignore[attr-defined]
            query_embeddings=[embedding],
            n_results=min(limit, available),
            include=["documents", "metadatas", "distances"],
        )
        documents = _first(response.get("documents"))
        metadatas = _first(response.get("metadatas"))
        distances = _first(response.get("distances"))
        results: list[SearchResult] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            if (
                not isinstance(document, str)
                or not isinstance(metadata, dict)
                or isinstance(distance, bool)
                or not isinstance(distance, int | float)
            ):
                continue
            url = metadata.get("url")
            results.append(
                SearchResult(
                    item_id=str(metadata["item_id"]),
                    index=int(metadata["chunk_index"]),
                    text=document,
                    title=str(metadata["title"]),
                    url=str(url) if url else None,
                    source_type=str(metadata["source_type"]),
                    distance=float(distance),
                )
            )
        return results

    def delete_item(self, item_id: str) -> None:
        self._get_collection().delete(where={"item_id": item_id})  # type: ignore[attr-defined]

    def count(self) -> int:
        return int(self._get_collection().count())  # type: ignore[attr-defined]

    def _get_collection(self) -> object:
        if self._collection is not None:
            return self._collection
        import chromadb

        client = chromadb.HttpClient(
            host=self._settings.chroma_host, port=self._settings.chroma_port
        )
        self._collection = client.get_or_create_collection(
            name=self._settings.chroma_collection,
            metadata={
                "hnsw:space": "cosine",
                "embedding_dimension": self._settings.embedding_dimensions,
            },
        )
        return self._collection


def _metadata(record: ChunkRecord) -> dict[str, str | int]:
    metadata: dict[str, str | int] = {
        "item_id": record.item_id,
        "chunk_index": record.index,
        "source_type": record.source_type,
        "title": record.title,
    }
    if record.url:
        metadata["url"] = record.url
    return metadata


def _first(values: object) -> list[object]:
    if isinstance(values, list) and values and isinstance(values[0], list):
        return list(values[0])
    return []
