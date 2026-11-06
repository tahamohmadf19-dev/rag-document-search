"""Small local UI: upload a PDF, ask a question, see the answer and its citations.

Run with: streamlit run streamlit_app.py  (needs: pip install -e .[ui])
"""
import tempfile
from pathlib import Path

import streamlit as st

from rag_core import answer_question, ingest_pdf
from vector_db import QdrantStorage


@st.cache_resource
def get_store() -> QdrantStorage:
    return QdrantStorage(location=":memory:")


st.title("RAG Document Search")
st.caption("Answers are extracted from your document. Nothing is sent to an external service.")

store = get_store()
uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded is not None and st.button("Index document"):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "document.pdf"
        path.write_bytes(uploaded.getvalue())
        try:
            count = ingest_pdf(str(path), store, source_id=uploaded.name)
            st.success(f"Indexed {count} chunks from {uploaded.name}.")
        except ValueError as error:
            st.error(str(error))

question = st.text_input("Question")
if question:
    try:
        answer = answer_question(question, store)
    except ValueError as error:
        st.error(str(error))
    else:
        st.write(answer.text)
        for citation in answer.citations:
            st.caption(f"{citation.source}, page {citation.page} (score {citation.score:.2f})")
