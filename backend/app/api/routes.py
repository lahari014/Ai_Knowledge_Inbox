from typing import cast

from fastapi import APIRouter, Request

from app.api.errors import AppError
from app.api.schemas import (
    CitationSchema,
    HealthResponse,
    IngestRequest,
    ItemListResponse,
    ItemResponse,
    ItemSummarySchema,
    QueryRequest,
    QueryResponse,
)
from app.core.config import Settings
from app.domain.models import Item, ItemSummary
from app.services.ingestion import IngestionService
from app.services.retrieval import QueryService

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    mode = _settings(request).mode
    if mode == "openai":
        return HealthResponse(status="ok", mode="openai")
    return HealthResponse(status="ok", mode="demo")


@router.post(
    "/ingest", response_model=ItemResponse, status_code=201, response_model_exclude_none=True
)
def ingest(body: IngestRequest, request: Request) -> ItemResponse:
    service = _ingestion(request)
    if body.type == "note":
        item = service.ingest_note(body.text, body.title)
    else:
        item = service.ingest_url(body.url)
    return ItemResponse(item=_summary_schema(_summary_from_item(item, _settings(request))))


@router.get("/items", response_model=ItemListResponse, response_model_exclude_none=True)
def list_items(request: Request) -> ItemListResponse:
    repository = request.app.state.repository
    return ItemListResponse(items=[_summary_schema(item) for item in repository.list_newest()])


@router.post("/query", response_model=QueryResponse, response_model_exclude_none=True)
def query(body: QueryRequest, request: Request) -> QueryResponse:
    settings = _settings(request)
    question = body.question.strip()
    if not question:
        raise AppError("VALIDATION_ERROR", "Question cannot be empty.", 422)
    if len(question) > settings.question_max_length:
        raise AppError("VALIDATION_ERROR", "Question exceeds the configured length limit.", 422)
    if body.limit is not None and not 1 <= body.limit <= settings.query_max_limit:
        raise AppError("VALIDATION_ERROR", "Result limit is outside the configured bounds.", 422)
    result = _query_service(request).ask(question, body.limit)
    mode = "openai" if result.mode == "openai" else "demo"
    return QueryResponse(
        answer=result.answer,
        mode=mode,
        citations=[
            CitationSchema(
                number=citation.number,
                item_id=citation.item_id,
                label=citation.label,
                url=citation.url,
                snippet=citation.snippet,
                score=citation.score,
            )
            for citation in result.citations
        ],
    )


def _settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def _ingestion(request: Request) -> IngestionService:
    return cast(IngestionService, request.app.state.ingestion)


def _query_service(request: Request) -> QueryService:
    return cast(QueryService, request.app.state.query_service)


def _summary_from_item(item: Item, settings: Settings) -> ItemSummary:
    from app.db.repository import build_preview

    return ItemSummary(
        id=item.id,
        source_type=item.source_type,
        title=item.title,
        url=item.url,
        preview=build_preview(item.raw_content, settings.preview_chars),
        created_at=item.created_at,
    )


def _summary_schema(item: ItemSummary) -> ItemSummarySchema:
    return ItemSummarySchema(
        id=item.id,
        source_type=item.source_type,
        title=item.title,
        url=item.url,
        preview=item.preview,
        created_at=item.created_at,
    )
