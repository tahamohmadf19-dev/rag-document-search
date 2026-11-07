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
# Share of the question's content words that must appear in the best passage. The default embeddings
# are lexical character n-grams, so an unrelated question can score above MIN_SCORE through shared
# letter patterns alone ("capital of Australia" against a billing policy). Requiring at least half of
# the question's own words in the passage rejects those. Chosen as "at least half" before measuring,
# not tuned on the evaluation set.
MIN_TERM_COVERAGE = 0.5
MAX_TOP_K = 20
ABSTAIN_MESSAGE = "I do not know based on the indexed documents."
_SENTENCE_END = re.compile(r"(?<=[.!?؟])\s+")
_WORD = re.compile(r"\w+", re.UNICODE)
_ARABIC_MARKS = re.compile(r"[ؐ-ًؚ-ٰٟ]")
_STOPWORDS = frozenset(
    """a an the is are was were be been do does did can could should must may might will would
    what which who whom whose when where why how many much long often of in on at to for from by with
    about into than then this that these those it its i me my we our you your he she they them their
    and or not no if there here have has had any some all each
    ما ماذا من متى أين اين كيف كم هل لماذا هو هي هم هذا هذه ذلك تلك في على عن إلى الى مع أو او و
    ثم قد لا لم لن ان أن إن كان كانت يمكن التي الذي الذين""".split()
)


def _content_terms(text: str) -> list[str]:
    words = (_ARABIC_MARKS.sub("", w).lower() for w in _WORD.findall(text))
    return [w for w in words if len(w) >= 3 and w not in _STOPWORDS and not w.isdigit()]


def _same_word(a: str, b: str) -> bool:
    # Light matching across word forms: expire/expires, submit/submitted, استعارته/استعارة.
    return a == b or (len(a) >= 4 and len(b) >= 4 and a[:4] == b[:4])


def term_coverage(question: str, passage: str) -> float:
    """Share of the question's content words that appear in the passage (1.0 if it has none)."""
    wanted = set(_content_terms(question))
    if not wanted:
        return 1.0
    present = _content_terms(passage)
    return sum(any(_same_word(w, p) for p in present) for w in wanted) / len(wanted)


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


def answer_question(
    question: str,
    store: QdrantStorage,
    top_k: int = 5,
    min_score: float = MIN_SCORE,
    min_coverage: float = MIN_TERM_COVERAGE,
) -> Answer:
    found = retrieve(question, store, top_k)
    if not found["scores"] or found["scores"][0] < min_score:
        return Answer(ABSTAIN_MESSAGE, abstained=True)
    best = found["contexts"][0]
    if term_coverage(question, best) < min_coverage:
        return Answer(ABSTAIN_MESSAGE, abstained=True)
    citations = [
        Citation(source, page, score)
        for source, page, score in zip(found["sources"], found["pages"], found["scores"])
        if score >= min_score
    ]
    return Answer(_best_sentences(question, best), abstained=False, citations=citations)
