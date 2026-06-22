"""RAG agent backed by a persistent Chroma index of the Transformer paper."""

from __future__ import annotations

import argparse
import hashlib
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

from config import CHROMA_DIR, PDF_PATH, SETTINGS, get_embeddings
from state import AgentState, Source


def _pdf_fingerprint() -> str:
    digest = hashlib.sha256()
    with PDF_PATH.open("rb") as pdf_file:
        for block in iter(lambda: pdf_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"Knowledge-base PDF not found: {PDF_PATH}")

    fingerprint = _pdf_fingerprint()
    store = Chroma(
        collection_name=f"attention_paper_{fingerprint[:12]}",
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )
    if not store.get(limit=1)["ids"]:
        pages = PyPDFLoader(str(PDF_PATH)).load()
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=SETTINGS.rag.chunk_size,
            chunk_overlap=SETTINGS.rag.chunk_overlap,
        ).split_documents(pages)
        ids = [
            hashlib.sha256(
                f'{fingerprint}:{doc.metadata.get("page", 0)}:{index}:{doc.page_content}'.encode()
            ).hexdigest()
            for index, doc in enumerate(chunks)
        ]
        store.add_documents(chunks, ids=ids)
    return store


def rag_agent_node(state: AgentState) -> dict:
    matches = get_vector_store().similarity_search_with_relevance_scores(
        state["query"], k=SETTINGS.rag.top_k
    )
    context_parts: list[str] = []
    sources: list[Source] = list(state.get("sources", []))
    known_urls = {source["url"] for source in sources}
    seen_pages: set[int] = set()
    for document, score in matches:
        page = int(document.metadata.get("page", 0)) + 1
        context_parts.append(
            f"[Attention Is All You Need, p. {page}; relevance={score:.3f}]\n"
            f"{document.page_content}"
        )
        page_url = f"{PDF_PATH.name}#page={page}"
        if page not in seen_pages and page_url not in known_urls:
            seen_pages.add(page)
            sources.append(
                {
                    "title": f"Attention Is All You Need, page {page}",
                    "url": page_url,
                }
            )
            known_urls.add(page_url)
    existing = state.get("gathered_context", "")
    rag_context = "\n\n".join(context_parts)
    return {
        "gathered_context": f"{existing}\n\n[PDF retrieval]\n{rag_context}".strip(),
        "sources": sources,
    }


def build_rag_graph():
    builder = StateGraph(AgentState)
    builder.add_node("rag_agent", rag_agent_node)
    builder.add_edge(START, "rag_agent")
    builder.add_edge("rag_agent", END)
    return builder.compile()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run only the RAG retrieval agent")
    parser.add_argument("query", nargs="+", help="Question about the Transformer paper")
    args = parser.parse_args()
    query = " ".join(args.query)
    result = build_rag_graph().invoke(
        {"query": query, "messages": [HumanMessage(content=query)]}
    )
    print(result.get("gathered_context", "No relevant passages found."))


if __name__ == "__main__":
    main()
