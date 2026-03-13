# RAG Document Search with Qdrant and FLAN‑T5

RAG Document Search is a proof‑of‑concept application that ingests PDF documents, indexes them into a vector store and allows you to ask natural‑language questions over their contents. It combines retrieval‑augmented generation (RAG) with a lightweight API server and a simple web interface.

Modern large language models aren’t trained on your proprietary data. RAG adds your data to the LLM’s knowledge base so the model can ground its answers in the relevant context. When a question arrives, the system retrieves the most similar chunks from your documents and feeds them to the language model along with the prompt to produce a grounded answer.

## Features

- **Ingestion pipeline**: Loads PDF files, splits them into chunks and generates embeddings via sentence‑transformers.
- **Vector database**: Uses Qdrant to store embeddings and perform similarity search.
- **FastAPI + Inngest**: Provides event‑driven endpoints for ingesting documents and querying them.
- **FLAN‑T5 model**: Generates answers conditioned on the retrieved context.
- **Streamlit UI** (coming soon): An easy interface for uploading documents and asking questions.

## Getting Started

1. Install Qdrant (e.g. via Docker) and create a virtual environment.
2. Copy `.env.example` to `.env` and configure `QDRANT_HOST`, `QDRANT_PORT`, `FLAN_MODEL`, and `EMBEDDING_MODEL`.
3. Install dependencies from `pyproject.toml`, then start the API server:

```
uvicorn rag_document_search.api:app --host 0.0.0.0 --port 8000 --reload
```

4. Ingest a PDF and ask questions by sending POST requests to `/ingest` and `/query`.

See the full project documentation for examples, API usage, proposed directory structure and roadmap.
