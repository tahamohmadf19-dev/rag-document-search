"""The RAG contract: ingest, retrieve, answer with citations, or abstain.

The default answer step is extractive: it returns sentences from retrieved chunks, so it cannot
invent facts. Generation with a language model is optional and lives in main.py.
"""
import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

from data_loader import chunk_text, embed_texts, load_pdf_chunks
from vector_db import QdrantStorage

MIN_SCORE = 0.16
MAX_TOP_K = 20
ABSTAIN_MESSAGE = "I do not know based on the indexed documents."
_SENTENCE_END = re.compile(r"(?<=[.!?؟])\s+")


@dataclass(frozen=True)
class Citation:
    source: str
    page: Optional[int]
    score: float


@dataclass(frozen=True)
class Answer:
    text: str
    abstained: bool
    citations: list[Citation] = field(default_factory=list)


def _point_id(source_id: str, index: int) -> str:
    # A stable id makes re-ingesting a document an update, not a duplicate.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{index}"))


def ingest_pdf(pdf_path: str, store: QdrantStorage, source_id: Optional[str] = None) -> int:
    source = source_id or pdf_path
    chunks = load_pdf_chunks(pdf_path)
    if not chunks:
        raise ValueError(f"No extractable text in {pdf_path}. Scanned PDFs need OCR first.")
    texts = [chunk.text for chunk in chunks]
    return store.upsert(
        [_point_id(source, i) for i in range(len(chunks))],
        embed_texts(texts),
        [{"source": source, "page": chunk.page, "text": chunk.text} for chunk in chunks],
    )


def ingest_text(text: str, source_id: str, store: QdrantStorage, page: int = 1) -> int:
    chunks = chunk_text(text)
    return store.upsert(
        [_point_id(f"{source_id}:{page}", i) for i in range(len(chunks))],
        embed_texts(chunks),
        [{"source": source_id, "page": page, "text": chunk} for chunk in chunks],
    )


def validate_query(question: str, top_k: int) -> str:
    cleaned = (question or "").strip()
    if not cleaned:
        raise ValueError("Question is empty.")
    if len(cleaned) > 2000:
        raise ValueError("Question is too long.")
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}.")
    return cleaned


def retrieve(question: str, store: QdrantStorage, top_k: int = 5) -> dict[str, list]:
    cleaned = validate_query(question, top_k)
    return store.search(embed_texts([cleaned])[0], top_k)


def _best_sentences(question: str, context: str, count: int = 2) -> str:
    sentences = [s.strip() for s in _SENTENCE_END.split(context) if s.strip()]
    if len(sentences) <= count:
        return " ".join(sentences)
    vectors = embed_texts([question] + sentences)
    query = vectors[0]
    scores = [sum(a * b for a, b in zip(query, vector)) for vector in vectors[1:]]
    ranked = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:count]
    return " ".join(sentences[i] for i in sorted(ranked))


def answer_question(question: str, store: QdrantStorage, top_k: int = 5, min_score: float = MIN_SCORE) -> Answer:
    found = retrieve(question, store, top_k)
    if not found["scores"] or found["scores"][0] < min_score:
        return Answer(ABSTAIN_MESSAGE, abstained=True)
    best = found["contexts"][0]
    citations = [
        Citation(source, page, score)
        for source, page, score in zip(found["sources"], found["pages"], found["scores"])
        if score >= min_score
    ]
    return Answer(_best_sentences(question, best), abstained=False, citations=citations)
