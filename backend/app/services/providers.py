from dataclasses import dataclass

from app.core.config import Settings
from app.domain.protocols import AnswerProvider, EmbeddingProvider
from app.services.answering import DemoAnswerProvider, OpenAIAnswerProvider
from app.services.embeddings import DemoEmbeddingProvider, OpenAIEmbeddingProvider


@dataclass(frozen=True)
class AiProviders:
    embeddings: EmbeddingProvider
    answers: AnswerProvider
    mode: str


def build_ai_providers(settings: Settings) -> AiProviders:
    if settings.mode == "openai":
        return AiProviders(
            embeddings=OpenAIEmbeddingProvider(settings),
            answers=OpenAIAnswerProvider(settings),
            mode="openai",
        )
    return AiProviders(
        embeddings=DemoEmbeddingProvider(settings.demo_embedding_dimensions),
        answers=DemoAnswerProvider(),
        mode="demo",
    )
