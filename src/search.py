import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "llama-3.1-8b-instant"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from src.data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        groq_api_key = os.getenv("GROQ_API_KEY", "")
        if not groq_api_key:
            raise ValueError("Missing GROQ_API_KEY. Add it to your .env file before running RAG summarization.")
        self.llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    @staticmethod
    def _format_source(metadata: dict) -> str:
        source = os.path.basename(metadata.get("source", "unknown source"))
        page = metadata.get("page")
        if isinstance(page, int):
            return f"{source} (page {page + 1})"
        return source

    def retrieve(self, query: str, top_k: int = 5):
        return self.vectorstore.query(query, top_k=top_k)

    def answer_question(self, query: str, top_k: int = 5) -> dict:
        results = self.retrieve(query, top_k=top_k)
        contexts = []
        sources = []

        for result in results:
            metadata = result.get("metadata") or {}
            text = metadata.get("text", "").strip()
            if not text:
                continue
            source_label = self._format_source(metadata)
            contexts.append(f"Source: {source_label}\n{text}")
            sources.append(
                {
                    "source": metadata.get("source", "unknown source"),
                    "page": metadata.get("page"),
                    "label": source_label,
                    "text": text,
                    "distance": result.get("distance"),
                }
            )

        if not contexts:
            return {"answer": "No relevant documents found.", "sources": []}

        prompt = (
            "Answer the user's question using only the provided context. "
            "If the answer is not present in the context, say that clearly.\n\n"
            f"Question: {query}\n\n"
            "Context:\n"
            + "\n\n".join(contexts)
            + "\n\nAnswer:"
        )
        response = self.llm.invoke([prompt])
        return {"answer": response.content, "sources": sources}

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        return self.answer_question(query, top_k=top_k)["answer"]

# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
