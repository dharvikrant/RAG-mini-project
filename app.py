
import hashlib
import os
from pathlib import Path

import streamlit as st

from src.data_loader import list_supported_files
from src.search import RAGSearch
from src.vectorstore import FaissVectorStore
from src.data_loader import load_documents_from_files


st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
)


def build_persist_dir(selected_files):
    digest_input = "|".join(str(path) for path in selected_files)
    digest = hashlib.md5(digest_input.encode("utf-8")).hexdigest()[:12]
    return os.path.join("faiss_store", digest)


def ensure_vectorstore(persist_dir, selected_files):
    store = FaissVectorStore(persist_dir)

    faiss_path = os.path.join(persist_dir, "faiss.index")
    meta_path = os.path.join(persist_dir, "metadata.pkl")

    if os.path.exists(faiss_path) and os.path.exists(meta_path):
        store.load()
    else:
        docs = load_documents_from_files(selected_files)

        if not docs:
            raise ValueError(
                "No readable content found in the selected files."
            )

        store.build_from_documents(docs)

    return store


@st.cache_resource(show_spinner="Loading documents and AI model...")
def get_rag_search(selected_paths):
    selected_files = [Path(path) for path in selected_paths]
    persist_dir = build_persist_dir(selected_files)

    ensure_vectorstore(persist_dir, selected_files)

    return RAGSearch(persist_dir=persist_dir)


st.title("📚 RAG Document Assistant")
st.write(
    "Upload-free document question answering: "
    "select a PDF from your data folder and ask questions about it."
)

st.info(
    "Choose documents below, then enter a question. "
    "Answers can include the source document and page numbers."
)

try:
    available_files = list_supported_files("data")
except Exception as error:
    st.error(f"Could not read the data folder: {error}")
    st.stop()

if not available_files:
    st.warning(
        "No supported documents found. Add your PDF files "
        "to the project's data folder."
    )
    st.stop()

file_options = [str(path) for path in available_files]

selected_paths = st.multiselect(
    "Select document(s)",
    options=file_options,
    format_func=lambda path: Path(path).name,
    placeholder="Choose one or more documents",
)

question = st.text_input(
    "Your question",
    placeholder="Example: What is the main topic of this document?",
)

if st.button("Get Answer", type="primary"):
    if not selected_paths:
        st.warning("Please select at least one document.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Searching documents and generating your answer..."):
                rag_search = get_rag_search(tuple(selected_paths))
                result = rag_search.answer_question(
                    question.strip(),
                    top_k=3,
                )

            st.subheader("Answer")
            st.write(result["answer"])

            sources = result.get("sources", [])
            if sources:
                st.subheader("Sources")
                for source in sources:
                    st.write(f"- {source['label']}")

        except Exception as error:
            st.error(f"Something went wrong: {error}")
            st.caption(
                "Check the Command Prompt for the full error details."
            )
