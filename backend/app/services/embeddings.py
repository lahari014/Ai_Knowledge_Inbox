import hashlib
import math
import re

from openai import APIError, APITimeoutError, OpenAI

from app.api.errors import AppError
from app.core.config import Settings

_TOKEN = re.compile(r"[a-z0-9]+")


class DemoEmbeddingProvider:
    """Stable hashed embeddings so the full pipeline runs without an API key.

    This is a feature-hashing vector, not a neural embedding. It is deterministic and
    good enough to retrieve overlapping vocabulary. It is not a substitute for OpenAI
    retrieval quality.
    """

    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_normalize(_hash_features(text, self.dimensions)) for text in texts]


class OpenAIEmbeddingProvider:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = OpenAI(
            api_key=settings.openai_api_key, timeout=settings.fetch_timeout_seconds
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self._client.embeddings.create(
                model=self._settings.openai_embedding_model,
                input=texts,
            )
        except APITimeoutError as exc:
            raise AppError("UPSTREAM_TIMEOUT", "The embedding provider timed out.", 504) from exc
        except APIError as exc:
            raise AppError(
                "PROVIDER_UNAVAILABLE", "The embedding provider is unavailable.", 503
            ) from exc
        return [list(item.embedding) for item in response.data]


def _hash_features(text: str, dimensions: int) -> list[float]:
    vector = [0.0] * dimensions
    tokens = _TOKEN.findall(text.lower())
    features = tokens + [f"{left}_{right}" for left, right in zip(tokens, tokens[1:])]
    for feature in features:
        digest = hashlib.sha256(feature.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign
    return vector


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]
