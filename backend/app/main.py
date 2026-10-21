from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.errors import AppError, error_response
from app.api.middleware import RequestIdMiddleware, request_id_var
from app.api.routes import router
from app.core.config import Settings
from app.core.logging import configure_logging, get_logger
from app.db.repository import ItemRepository
from app.domain.protocols import AnswerProvider, EmbeddingProvider, PageFetcher, VectorStore
from app.services.chunking import Chunker
from app.services.fetching import ContentFetcher
from app.services.ingestion import IngestionService
from app.services.providers import build_ai_providers
from app.services.retrieval import QueryService
from app.services.vector_store import ChromaVectorStore

logger = get_logger("app.main")


def create_app(
    settings: Settings | None = None,
    *,
    fetcher: PageFetcher | None = None,
    vector_store: VectorStore | None = None,
    embeddings: EmbeddingProvider | None = None,
    answers: AnswerProvider | None = None,
) -> FastAPI:
    runtime_settings = settings or Settings()
    configure_logging(runtime_settings.log_level)
    repository = ItemRepository(runtime_settings)
    providers = build_ai_providers(runtime_settings)
    selected_embeddings = embeddings or providers.embeddings
    selected_answers = answers or providers.answers
    selected_store = vector_store or ChromaVectorStore(runtime_settings)
    selected_fetcher = fetcher or ContentFetcher(runtime_settings)
    ingestion = IngestionService(
        runtime_settings,
        repository,
        Chunker(runtime_settings.chunk_size, runtime_settings.chunk_overlap),
        selected_embeddings,
        selected_store,
        selected_fetcher,
    )
    query_service = QueryService(
        selected_embeddings,
        selected_store,
        selected_answers,
        providers.mode if embeddings is None else runtime_settings.mode,
        runtime_settings.query_default_limit,
        runtime_settings.minimum_relevance_score,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        repository.create_schema()
        selected_store.ensure_ready()
        logger.info("application_started", mode=runtime_settings.mode, stage="startup")
        yield

    app = FastAPI(
        title="AI Knowledge Inbox",
        version="1.0.0",
        summary="Save notes and web pages, then ask questions with cited evidence.",
        lifespan=lifespan,
    )
    app.state.settings = runtime_settings
    app.state.repository = repository
    app.state.ingestion = ingestion
    app.state.query_service = query_service
    app.state.vector_store = selected_store
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[runtime_settings.cors_origin],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        expose_headers=["x-request-id"],
    )
    app.add_middleware(RequestIdMiddleware)
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(RequestValidationError, validation_handler)
    app.add_exception_handler(StarletteHTTPException, http_handler)
    app.add_exception_handler(Exception, unexpected_handler)
    app.include_router(router)
    return app


def app_error_handler(request: Request, exc: Exception) -> JSONResponse:
    error = (
        exc
        if isinstance(exc, AppError)
        else AppError("INTERNAL_ERROR", "An unexpected error occurred.", 500)
    )
    return _with_request_id(
        error_response(error.code, error.message, error.status_code, _request_id(request)), request
    )


def validation_handler(request: Request, exc: Exception) -> JSONResponse:
    message = "The request is invalid."
    if isinstance(exc, RequestValidationError) and exc.errors():
        detail = exc.errors()[0].get("msg", "")
        if isinstance(detail, str) and detail:
            message = detail.removeprefix("Value error, ")
    return _with_request_id(
        error_response("VALIDATION_ERROR", message, 422, _request_id(request)), request
    )


def http_handler(request: Request, exc: Exception) -> JSONResponse:
    status_code = exc.status_code if isinstance(exc, StarletteHTTPException) else 500
    if status_code == 404:
        response = error_response(
            "NOT_FOUND", "The requested resource does not exist.", 404, _request_id(request)
        )
    else:
        response = error_response(
            "HTTP_ERROR", "The request could not be completed.", status_code, _request_id(request)
        )
    return _with_request_id(response, request)


def unexpected_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("unhandled_error", error_type=type(exc).__name__, stage="failed")
    return _with_request_id(
        error_response(
            "INTERNAL_ERROR", "An unexpected error occurred.", 500, _request_id(request)
        ),
        request,
    )


def _request_id(request: Request) -> str:
    state_value = getattr(request.state, "request_id", "")
    if isinstance(state_value, str) and state_value:
        return state_value
    return request_id_var.get()


def _with_request_id(response: JSONResponse, request: Request) -> JSONResponse:
    response.headers["x-request-id"] = _request_id(request)
    return response
