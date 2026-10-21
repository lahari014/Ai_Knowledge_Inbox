from collections.abc import Iterator
from pathlib import Path

import pytest
from app.core.config import Settings
from app.main import create_app
from fastapi.testclient import TestClient
from tests.fakes import MemoryVectorStore


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        database_url=f"sqlite:///{tmp_path}/inbox.db",
        openai_api_key="",
        chroma_host="unused",
        _env_file=None,  # type: ignore[call-arg]
    )


@pytest.fixture
def vector_store() -> MemoryVectorStore:
    return MemoryVectorStore()


@pytest.fixture
def client(settings: Settings, vector_store: MemoryVectorStore) -> Iterator[TestClient]:
    app = create_app(settings, vector_store=vector_store)
    with TestClient(app) as test_client:
        yield test_client
