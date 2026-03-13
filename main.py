import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import uuid
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGChunkAndSrc, RAGSearchResult, RAGUpsertResult
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

# موديل مجاني محلي للإجابة
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
qa_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")


@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx: inngest.Context):

    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id

        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]

        QdrantStorage().upsert(ids, vecs, payloads)

        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run(
        "load-and-chunk",
        lambda: _load(ctx),
        output_type=RAGChunkAndSrc
    )

    ingested = await ctx.step.run(
        "embed-and-upsert",
        lambda: _upsert(chunks_and_src),
        output_type=RAGUpsertResult
    )

    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):

    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k)

        return RAGSearchResult(
            contexts=found["contexts"],
            sources=found["sources"]
        )

    def _answer(question: str, found: RAGSearchResult):
        context_block = "\n\n".join(f"- {c}" for c in found.contexts)

        prompt = (
            "You are a helpful assistant. Answer using only the context below.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n\n"
            "Write a complete answer in clear sentences. "
            "If the answer is not explicitly in the context, say: I do not know."
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )

        outputs = qa_model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        return {
            "answer": answer,
            "sources": found.sources,
            "num_contexts": len(found.contexts)
        }

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run(
        "embed-and-search",
        lambda: _search(question, top_k),
        output_type=RAGSearchResult
    )

    result = await ctx.step.run(
        "generate-answer",
        lambda: _answer(question, found)
    )

    return result


app = FastAPI()

inngest.fast_api.serve(
    app,
    inngest_client,
    functions=[rag_ingest_pdf, rag_query_pdf_ai]
)