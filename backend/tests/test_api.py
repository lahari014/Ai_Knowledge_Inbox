import logging
from datetime import datetime, timezone
from pathlib import Path

import pytest
from app.core.config import Settings
from app.db.repository import ItemRepository
from app.domain.models import FetchedContent, Item, SourceType
from app.main import create_app
from app.services.answering import INSUFFICIENT_ANSWER
from fastapi.testclient import TestClient
from tests.fakes import FailingVectorStore, MemoryVectorStore


def test_health_reports_openai_mode_when_a_key_is_configured(tmp_path: Path) -> None:
    configured = Settings(
        database_url=f"sqlite:///{tmp_path}/inbox.db",
        openai_api_key="test-key",
        chroma_host="unused",
        _env_file=None,  # type: ignore[call-arg]
    )
    with TestClient(create_app(configured, vector_store=MemoryVectorStore())) as test_client:
        response = test_client.get("/health")

    assert response.status_code == 200
    assert response.json()["mode"] == "openai"


def test_health_reports_demo_mode(client: TestClient) -> None:
    response = client.get("/health", headers={"x-request-id": "req-123"})

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "mode": "demo"}
    assert response.headers["x-request-id"] == "req-123"


def test_invalid_request_id_is_replaced(client: TestClient) -> None:
    response = client.get("/health", headers={"x-request-id": "bad id"})

    assert response.status_code == 200
    assert response.headers["x-request-id"] != "bad id"


def test_app_error_uses_stable_envelope(
    settings: Settings, vector_store: MemoryVectorStore
) -> None:
    from app.api.errors import AppError

    app = create_app(settings, vector_store=vector_store)

    @app.get("/boom")
    def boom() -> None:
        raise AppError("BAD_INPUT", "Invalid input", 422)

    with TestClient(app) as test_client:
        response = test_client.get("/boom")

    assert response.status_code == 422
    assert response.json() == {
        "error": {
            "code": "BAD_INPUT",
            "message": "Invalid input",
            "requestId": response.headers["x-request-id"],
        }
    }


def test_items_are_empty_before_ingestion(client: TestClient) -> None:
    response = client.get("/items")

    assert response.status_code == 200
    assert response.json() == {"items": []}


def test_repository_lists_newest_item_first(settings: Settings) -> None:
    repository = ItemRepository(settings)
    repository.create_schema()
    older = _item("older", datetime(2026, 10, 20, tzinfo=timezone.utc))
    newer = _item("newer", datetime(2026, 10, 21, tzinfo=timezone.utc))
    repository.add(older)
    repository.add(newer)

    listed = repository.list_newest()

    assert [item.id for item in listed] == ["newer", "older"]


def test_note_ingestion_returns_summary_and_preview(client: TestClient) -> None:
    text = "a" * 181
    response = client.post("/ingest", json={"type": "note", "text": text, "title": "Launch notes"})

    assert response.status_code == 201
    item = response.json()["item"]
    assert item["sourceType"] == "note"
    assert item["title"] == "Launch notes"
    assert item["preview"] == f"{'a' * 180}…"
    assert "url" not in item
    listed = client.get("/items")
    assert listed.json()["items"][0]["id"] == item["id"]


def test_short_preview_is_not_truncated(client: TestClient) -> None:
    response = client.post("/ingest", json={"type": "note", "text": "Short note"})

    assert response.status_code == 201
    assert response.json()["item"]["preview"] == "Short note"


def test_invalid_ingest_shapes_are_rejected(client: TestClient) -> None:
    responses = [
        client.post("/ingest", json={"text": "missing type"}),
        client.post("/ingest", json={"type": "file", "text": "nope"}),
        client.post("/ingest", json={"type": "note", "text": "   "}),
        client.post("/ingest", json={"type": "url", "url": "ftp://example.com/file"}),
    ]

    assert [response.status_code for response in responses] == [422, 422, 422, 422]
    assert client.get("/items").json()["items"] == []


def test_content_over_limit_is_rejected(settings: Settings) -> None:
    limited = settings.model_copy(update={"content_max_chars": 5})
    with TestClient(create_app(limited, vector_store=MemoryVectorStore())) as test_client:
        response = test_client.post("/ingest", json={"type": "note", "text": "123456"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_url_ingestion_uses_fetched_text(
    settings: Settings, vector_store: MemoryVectorStore
) -> None:
    app = create_app(settings, vector_store=vector_store, fetcher=FakeFetcher())
    with TestClient(app) as test_client:
        response = test_client.post(
            "/ingest", json={"type": "url", "url": "https://example.com/docs"}
        )

    assert response.status_code == 201
    item = response.json()["item"]
    assert item["sourceType"] == "url"
    assert item["title"] == "Example"
    assert item["url"] == "https://example.com/docs"
    assert item["preview"] == "Page body about rockets."
    assert vector_store.records[0].text == "Page body about rockets."
    assert vector_store.records[0].item_id == item["id"]


def test_failed_indexing_removes_the_item(settings: Settings) -> None:
    store = FailingVectorStore()
    with TestClient(create_app(settings, vector_store=store)) as test_client:
        response = test_client.post("/ingest", json={"type": "note", "text": "Keep nothing"})
        listed = test_client.get("/items")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "INDEX_UNAVAILABLE"
    assert store.deleted
    assert listed.json()["items"] == []


def test_query_without_knowledge_explains_the_gap(client: TestClient) -> None:
    response = client.post("/query", json={"question": "When is launch?"})

    assert response.status_code == 200
    assert response.json() == {"answer": INSUFFICIENT_ANSWER, "mode": "demo", "citations": []}


def test_query_rejects_empty_long_and_out_of_range_limits(client: TestClient) -> None:
    empty = client.post("/query", json={"question": "   "})
    too_long = client.post("/query", json={"question": "a" * 2001})
    too_many = client.post("/query", json={"question": "When?", "limit": 11})
    too_few = client.post("/query", json={"question": "When?", "limit": 0})

    assert [response.status_code for response in (empty, too_long, too_many, too_few)] == [
        422,
        422,
        422,
        422,
    ]
    assert {
        response.json()["error"]["code"] for response in (empty, too_long, too_many, too_few)
    } == {"VALIDATION_ERROR"}


def test_query_returns_bounded_citations(settings: Settings) -> None:
    compact = settings.model_copy(
        update={"chunk_size": 30, "chunk_overlap": 0, "query_default_limit": 5}
    )
    store = MemoryVectorStore()
    with TestClient(create_app(compact, vector_store=store)) as test_client:
        created = test_client.post(
            "/ingest", json={"type": "note", "text": "word " * 80, "title": "Words"}
        )
        default = test_client.post("/query", json={"question": "word"})
        limited = test_client.post("/query", json={"question": "word", "limit": 2})

    assert created.status_code == 201
    assert len(default.json()["citations"]) == 5
    assert len(limited.json()["citations"]) == 2
    first = default.json()["citations"][0]
    assert first["label"] == "Words"
    assert first["itemId"] == created.json()["item"]["id"]
    assert first["snippet"]
    assert 0 <= first["score"] <= 1
    assert first["number"] == 1


def test_request_log_excludes_saved_content(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    secret = "SUPER_SECRET_NOTE_CONTENT"
    with caplog.at_level(logging.INFO):
        response = client.post("/ingest", json={"type": "note", "text": secret, "title": "Secret"})

    assert response.status_code == 201
    assert secret not in caplog.text
    assert "request_completed" in caplog.text
    assert "/ingest" in caplog.text


def _item(item_id: str, created_at: datetime) -> Item:
    return Item(
        id=item_id,
        source_type=SourceType.NOTE,
        title=item_id,
        url=None,
        raw_content=item_id,
        created_at=created_at,
    )


class FakeFetcher:
    def fetch(self, url: str) -> FetchedContent:
        return FetchedContent(url=url, title="Example", text="Page body about rockets.")
