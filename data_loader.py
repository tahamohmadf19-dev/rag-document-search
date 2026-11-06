"""PDF loading, chunking and embeddings."""
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import HashingVectorizer

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
EMBEDDING_DIM = 2048
_SENTENCE_END = re.compile(r"(?<=[.!?؟])\s+")


@dataclass(frozen=True)
class Chunk:
    text: str
    page: int


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text at sentence boundaries, carrying some overlap between chunks."""
    if overlap >= size:
        raise ValueError("Overlap must be smaller than chunk size.")
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return []
    sentences = _SENTENCE_END.split(cleaned)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        # A single very long sentence is cut by length.
        while len(sentence) > size:
            room = size - len(current) - (1 if current else 0)
            head, sentence = sentence[:room], sentence[room:]
            chunks.append(f"{current} {head}".strip())
            current = chunks[-1][-overlap:] if overlap else ""
        if len(current) + len(sentence) + 1 > size and current:
            chunks.append(current.strip())
            current = current[-overlap:] if overlap else ""
        current = f"{current} {sentence}".strip()
    if current.strip():
        chunks.append(current.strip())
    return chunks


def load_pdf_chunks(pdf_path: str) -> list[Chunk]:
    """Read a PDF page by page, so each chunk can cite its page number."""
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    chunks: list[Chunk] = []
    for number, page in enumerate(PdfReader(str(path)).pages, start=1):
        for text in chunk_text(page.extract_text() or ""):
            chunks.append(Chunk(text=text, page=number))
    return chunks


def load_and_chunk_pdf(pdf_path: str) -> list[str]:
    return [chunk.text for chunk in load_pdf_chunks(pdf_path)]


_hashing = HashingVectorizer(
    n_features=EMBEDDING_DIM,
    analyzer="char_wb",
    ngram_range=(3, 5),
    alternate_sign=False,
    norm="l2",
)


def _embed_hashing(texts: Sequence[str]) -> list[list[float]]:
    """Local character n-gram vectors. Lexical similarity, not semantic meaning.

    This default needs no model download and gives the same output every run, which makes
    tests and the evaluation reproducible.
    """
    return _hashing.transform(list(texts)).toarray().astype(np.float32).tolist()


def _embed_sentence_transformers(texts: Sequence[str]) -> list[list[float]]:
    """Dense embeddings. Optional: pip install -e .[semantic]"""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(os.environ["EMBEDDING_MODEL"])
    return model.encode(list(texts), normalize_embeddings=True).tolist()


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    if not texts:
        return []
    backend = os.getenv("EMBEDDING_BACKEND", "hashing")
    if backend == "hashing":
        return _embed_hashing(texts)
    if backend == "sentence-transformers":
        return _embed_sentence_transformers(texts)
    raise ValueError(f"Unknown EMBEDDING_BACKEND: {backend}")
