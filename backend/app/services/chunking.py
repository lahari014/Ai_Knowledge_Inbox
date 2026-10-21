import re

from app.domain.models import Chunk

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_BOUNDARY = re.compile(r"\n\s*\n")


class Chunker:
    """Split text into ordered chunks of at most `chunk_size` characters.

    Paragraphs and sentences stay intact when they fit. Oversized pieces are split on
    character boundaries. Each later chunk repeats at most `overlap` trailing characters
    from the previous chunk so a sentence cut by a boundary remains findable.
    """

    def __init__(self, chunk_size: int = 800, overlap: int = 120) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> list[Chunk]:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized:
            return []
        packed: list[str] = []
        current = ""
        for segment in self._segments(normalized):
            candidate = segment if not current else f"{current} {segment}"
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue
            if current:
                packed.append(current)
                current = self._with_overlap(current, segment)
            else:
                current = segment
        if current:
            packed.append(current)
        return [Chunk(index=index, text=part) for index, part in enumerate(packed)]

    def _with_overlap(self, previous: str, segment: str) -> str:
        if self.overlap == 0:
            return segment
        tail = previous[-self.overlap :]
        separator = "" if tail.endswith(" ") else " "
        combined = f"{tail}{separator}{segment}"
        if len(combined) <= self.chunk_size:
            return combined
        return segment

    def _segments(self, text: str) -> list[str]:
        paragraphs = [part.strip() for part in _PARAGRAPH_BOUNDARY.split(text) if part.strip()]
        segments: list[str] = []
        for paragraph in paragraphs:
            if len(paragraph) <= self.chunk_size:
                segments.append(paragraph)
                continue
            sentences = [
                part.strip() for part in _SENTENCE_BOUNDARY.split(paragraph) if part.strip()
            ]
            if not sentences:
                sentences = [paragraph]
            for sentence in sentences:
                segments.extend(self._hard_split(sentence))
        return segments

    def _hard_split(self, segment: str) -> list[str]:
        if len(segment) <= self.chunk_size:
            return [segment]
        return [
            segment[start : start + self.chunk_size]
            for start in range(0, len(segment), self.chunk_size)
        ]
