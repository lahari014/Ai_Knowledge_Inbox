from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven runtime settings. Secrets stay in the environment."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "AI Knowledge Inbox"
    database_url: str = "sqlite:///./data/inbox.db"
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection: str = "knowledge"
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_dimensions: int = 1536
    demo_embedding_dimensions: int = 384
    chunk_size: int = 800
    chunk_overlap: int = 120
    query_default_limit: int = 5
    query_max_limit: int = 10
    minimum_relevance_score: float = 0.15
    question_max_length: int = 2000
    content_max_chars: int = 100_000
    fetch_timeout_seconds: float = 10.0
    fetch_max_bytes: int = 2_000_000
    fetch_max_redirects: int = 3
    cors_origin: str = "http://localhost:5173"
    log_level: str = "INFO"
    preview_chars: int = Field(default=180, ge=1)

    @property
    def mode(self) -> str:
        return "openai" if self.openai_api_key.strip() else "demo"

    @property
    def embedding_dimensions(self) -> int:
        if self.mode == "openai":
            return self.openai_embedding_dimensions
        return self.demo_embedding_dimensions
