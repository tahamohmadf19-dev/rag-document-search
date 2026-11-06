"""FastAPI app serving event-driven ingest and query functions through Inngest.

Run a local Inngest dev server to send events to it. See the README.
The optional FLAN-T5 model loads only when a request asks for generation, so starting the
service stays fast and needs no model download.
"""
import logging
from functools import lru_cache
from pathlib import Path

import inngest
import inngest.fast_api
from dotenv import load_dotenv
from fastapi import FastAPI

from custom_types import RAGChunkAndSrc, RAGQueryResult, RAGSearchResult, RAGUpsertResult
from data_loader import embed_texts, load_pdf_chunks
from rag_core import ABSTAIN_MESSAGE, MIN_SCORE, _point_id, answer_question, validate_query
from vector_db import QdrantStorage

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)


@lru_cache(maxsize=1)
def _generator():
    """Load FLAN-T5 on first use. Needs the optional dependency group: pip install -e .[generate]"""
    import os

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    name = os.getenv("FLAN_MODEL", "google/flan-t5-large")
    return AutoTokenizer.from_pretrained(name), AutoModelForSeq2SeqLM.from_pretrained(name)


def generate_answer(question: str, contexts: list[str]) -> str:
    tokenizer, model = _generator()
    context_block = "\n\n".join(f"- {context}" for context in contexts)
    prompt = (
        "Answer using only the context below.\n\n"
        f"Context:\n{context_block}\n\nQuestion: {question}\n\n"
        "If the answer is not explicitly in the context, say: I do not know."
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


@inngest_client.create_function(
    fn_id="RAG: Ingest PDF", trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load() -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        if not Path(pdf_path).is_file():
            raise inngest.NonRetriableError(f"PDF not found: {pdf_path}")
        chunks = load_pdf_chunks(pdf_path)
        return RAGChunkAndSrc(
            chunks=[chunk.text for chunk in chunks],
            pages=[chunk.page for chunk in chunks],
            source_id=ctx.event.data.get("source_id", pdf_path),
        )

    def _upsert(loaded: RAGChunkAndSrc) -> RAGUpsertResult:
        count = len(loaded.chunks)
        QdrantStorage().upsert(
            [_point_id(loaded.source_id, i) for i in range(count)],
            embed_texts(loaded.chunks),
            [
                {"source": loaded.source_id, "page": loaded.pages[i], "text": loaded.chunks[i]}
                for i in range(count)
            ],
        )
        return RAGUpsertResult(ingested=count)

    loaded = await ctx.step.run("load-and-chunk", _load, output_type=RAGChunkAndSrc)
    result = await ctx.step.run("embed-and-upsert", lambda: _upsert(loaded), output_type=RAGUpsertResult)
    return result.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query PDF", trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))
    use_generation = bool(ctx.event.data.get("generate", False))

    def _search() -> RAGSearchResult:
        validate_query(question, top_k)
        store = QdrantStorage()
        found = store.search(embed_texts([question])[0], top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"], scores=found["scores"])

    def _answer(found: RAGSearchResult) -> RAGQueryResult:
        if not found.scores or found.scores[0] < MIN_SCORE:
            return RAGQueryResult(answer=ABSTAIN_MESSAGE, sources=[], num_contexts=0)
        if use_generation:
            text = generate_answer(question, found.contexts)
        else:
            text = answer_question(question, QdrantStorage(), top_k).text
        return RAGQueryResult(answer=text, sources=found.sources, num_contexts=len(found.contexts))

    found = await ctx.step.run("embed-and-search", _search, output_type=RAGSearchResult)
    result = await ctx.step.run("answer", lambda: _answer(found), output_type=RAGQueryResult)
    return result.model_dump()


app = FastAPI(title="RAG Document Search")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


inngest.fast_api.serve(app, inngest_client, functions=[rag_ingest_pdf, rag_query_pdf_ai])
