"""RAG contract tests: real Qdrant in embedded mode, real PDF parsing, no network or model download."""
import pytest
from fpdf import FPDF

from data_loader import chunk_text, embed_texts, load_and_chunk_pdf, load_pdf_chunks
from rag_core import (
    ABSTAIN_MESSAGE,
    MIN_SCORE,
    answer_question,
    ingest_pdf,
    ingest_text,
    retrieve,
    term_coverage,
    validate_query,
)
from vector_db import QdrantStorage


@pytest.fixture
def store():
    storage = QdrantStorage(location=":memory:", collection="test")
    yield storage
    storage.close()


@pytest.fixture
def pdf(tmp_path):
    document = FPDF()
    document.set_font("Helvetica", size=12)
    document.add_page()
    document.multi_cell(0, 8, "The warranty period is twenty four months from the date of purchase.")
    document.add_page()
    document.multi_cell(0, 8, "Customer support is open from nine in the morning until five in the evening.")
    path = tmp_path / "manual.pdf"
    document.output(str(path))
    return path


def test_chunks_respect_size_and_keep_all_words():
    text = " ".join(f"Sentence number {i} is here." for i in range(100))
    chunks = chunk_text(text, size=120, overlap=20)
    assert len(chunks) > 1
    assert all(len(chunk) <= 145 for chunk in chunks)
    joined = " ".join(chunks)
    assert all(f"number {i} " in joined for i in range(100))


def test_chunk_overlap_validation():
    with pytest.raises(ValueError):
        chunk_text("abc", size=10, overlap=10)


def test_empty_text_gives_no_chunks():
    assert chunk_text("   ") == []


def test_very_long_sentence_is_split():
    assert len(chunk_text("x" * 2000, size=300, overlap=30)) > 1


def test_pdf_chunks_keep_page_numbers(pdf):
    chunks = load_pdf_chunks(str(pdf))
    assert [chunk.page for chunk in chunks] == [1, 2]
    assert "warranty" in chunks[0].text
    assert load_and_chunk_pdf(str(pdf))[1].startswith("Customer support")


def test_missing_pdf_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="PDF not found"):
        load_pdf_chunks(str(tmp_path / "none.pdf"))


def test_embeddings_are_stable_and_normalized():
    first, second = embed_texts(["hello world"]), embed_texts(["hello world"])
    assert first == second
    assert abs(sum(v * v for v in first[0]) - 1.0) < 1e-5


def test_unknown_embedding_backend_is_rejected(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BACKEND", "nope")
    with pytest.raises(ValueError, match="Unknown"):
        embed_texts(["x"])


def test_end_to_end_pdf_answer_with_page_citation(pdf, store):
    assert ingest_pdf(str(pdf), store, source_id="manual") == 2
    answer = answer_question("How long is the warranty period?", store)
    assert not answer.abstained
    assert "twenty four months" in answer.text
    assert answer.citations[0].source == "manual" and answer.citations[0].page == 1


def test_reingesting_does_not_create_duplicates(pdf, store):
    ingest_pdf(str(pdf), store, source_id="manual")
    ingest_pdf(str(pdf), store, source_id="manual")
    assert store.client.count("test").count == 2


def test_empty_index_abstains(store):
    answer = answer_question("anything at all?", store)
    assert answer.abstained and answer.text == ABSTAIN_MESSAGE


def test_unrelated_question_abstains(store):
    ingest_text("Invoices are issued on the first business day of each month.", "billing", store)
    assert answer_question("How do volcanoes form?", store).abstained


@pytest.mark.parametrize(
    "question",
    # Both were answered with an unrelated passage before the lexical-support check:
    # their character n-grams scored above MIN_SCORE against the billing and handbook texts.
    ["What is the capital of Australia?", "Which programming language is the fastest for games?"],
)
def test_lexically_similar_but_unrelated_questions_abstain(store, question):
    from evaluation.evaluate import CORPUS

    for source, text in CORPUS.items():
        ingest_text(text, source, store)
    assert retrieve(question, store)["scores"][0] >= MIN_SCORE  # the score alone would have answered
    assert answer_question(question, store).abstained


def test_term_coverage_counts_question_words_found_in_the_passage():
    passage = "Unused leave expires at the end of March."
    assert term_coverage("When does unused leave expire?", passage) == 1.0
    assert term_coverage("What is the capital of Australia?", passage) == 0.0
    assert term_coverage("What is it?", passage) == 1.0  # only stopwords: nothing to check
    assert term_coverage("كم كتابًا يمكن للطالب استعارته؟", "يمكن للطالب استعارة خمسة كتب") >= 0.5


def test_answer_text_comes_from_the_document(store):
    source = "The late fee is two percent. Offices close at noon on Fridays. Parking is free."
    ingest_text(source, "doc", store)
    answer = answer_question("What is the late fee?", store)
    assert all(sentence.strip() in source for sentence in answer.text.split(". ") if sentence.strip())


def test_arabic_retrieval(store):
    ingest_text("مدة الإعارة أسبوعان قابلة للتجديد مرة واحدة.", "library", store)
    ingest_text("Invoices are issued every month.", "billing", store)
    assert retrieve("ما هي مدة الإعارة؟", store)["sources"][0] == "library"


@pytest.mark.parametrize(("question", "top_k"), [("", 5), ("   ", 5), ("x" * 2001, 5), ("ok", 0), ("ok", 21)])
def test_invalid_queries_are_rejected(question, top_k):
    with pytest.raises(ValueError):
        validate_query(question, top_k)


def test_upsert_rejects_mismatched_lengths(store):
    with pytest.raises(ValueError):
        store.upsert(["a"], [], [])


def test_scanned_pdf_gets_a_clear_error(tmp_path, store):
    document = FPDF()
    document.add_page()
    path = tmp_path / "blank.pdf"
    document.output(str(path))
    with pytest.raises(ValueError, match="No extractable text"):
        ingest_pdf(str(path), store)


def test_service_module_starts_without_loading_a_model():
    import sys

    import main

    assert "transformers" not in sys.modules
    assert any(getattr(route, "path", "") == "/health" for route in main.app.routes)


def test_evaluation_runs_and_reports_all_fields():
    from evaluation.evaluate import run

    results = run()
    assert results["answerable_questions"] == 14 and results["unanswerable_questions"] == 5
    assert 0.0 <= results["hit_at_1"] <= 1.0
    assert "Synthetic" in results["caveat"]
    assert results["false_answers_on_unanswerable"] == []
    assert results["questions_answered_incorrectly"] == []
