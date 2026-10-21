from app.domain.models import Citation, Context, QueryResult, SearchResult
from app.domain.protocols import AnswerProvider, EmbeddingProvider, VectorStore
from app.services.answering import INSUFFICIENT_ANSWER


class QueryService:
    def __init__(
        self,
        embeddings: EmbeddingProvider,
        vector_store: VectorStore,
        answers: AnswerProvider,
        mode: str,
        default_limit: int,
        minimum_score: float = 0.15,
    ) -> None:
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._answers = answers
        self._mode = mode
        self._default_limit = default_limit
        self._minimum_score = minimum_score

    def ask(self, question: str, limit: int | None = None) -> QueryResult:
        chosen_limit = self._default_limit if limit is None else limit
        if self._vector_store.count() == 0:
            self._embeddings.embed([question])
            return QueryResult(answer=INSUFFICIENT_ANSWER, mode=self._mode, citations=[])
        embedding = self._embeddings.embed([question])[0]
        matches = self._relevant(self._vector_store.query(embedding, chosen_limit))
        if not matches:
            return QueryResult(answer=INSUFFICIENT_ANSWER, mode=self._mode, citations=[])
        contexts = [
            Context(number=number, label=match.title, text=match.text)
            for number, match in enumerate(matches, start=1)
        ]
        return QueryResult(
            answer=self._answers.answer(question, contexts),
            mode=self._mode,
            citations=[to_citation(number, match) for number, match in enumerate(matches, start=1)],
        )

    def _relevant(self, matches: list[SearchResult]) -> list[SearchResult]:
        scored = [match for match in matches if to_score(match.distance) >= self._minimum_score]
        if not scored:
            return []
        strongest = to_score(scored[0].distance)
        return [match for match in scored if to_score(match.distance) >= strongest * 0.5]


def to_score(distance: float) -> float:
    return max(0.0, min(1.0, 1.0 - distance))


def to_citation(number: int, match: SearchResult) -> Citation:
    return Citation(
        number=number,
        item_id=match.item_id,
        label=match.title,
        url=match.url,
        snippet=match.text,
        score=to_score(match.distance),
    )
