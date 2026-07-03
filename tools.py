"""
tools.py

Local, offline tools used by the AI agents. Provides:
  - index_resumes: builds a LlamaIndex VectorStoreIndex backed by ChromaDB
    (in-memory / ephemeral) from raw resume text.
  - get_query_engine: returns a query engine over the built index.
  - resume_retrieval_tool: a CrewAI-compatible tool that agents call to
    retrieve relevant resume content for a given query.

No external network calls are made besides the local embedding model
download (HuggingFace) which happens once and is then cached locally.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import chromadb
from crewai.tools import tool
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state (module-level singletons for the lifetime of a single
# analysis request). These are reset via `reset_index()` before every
# new /analyze call so resumes from different requests never mix.
# ---------------------------------------------------------------------------

_embed_model: Optional[HuggingFaceEmbedding] = None
_chroma_client: Optional[chromadb.api.ClientAPI] = None
_index: Optional[VectorStoreIndex] = None
_query_engine: Optional[BaseQueryEngine] = None
_file_name_by_candidate: Dict[str, str] = {}


def _get_embed_model() -> HuggingFaceEmbedding:
    """Lazily instantiate and cache the HuggingFace embedding model."""
    global _embed_model
    if _embed_model is None:
        logger.info("Loading HuggingFace embedding model BAAI/bge-small-en-v1.5 ...")
        _embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        Settings.embed_model = _embed_model
        # Disable LlamaIndex's default LLM usage for indexing operations;
        # we only use LlamaIndex for retrieval, not generation.
        Settings.llm = None
    return _embed_model


def reset_index() -> None:
    """Reset all module-level index state. Call before indexing a new batch."""
    global _chroma_client, _index, _query_engine, _file_name_by_candidate
    _chroma_client = None
    _index = None
    _query_engine = None
    _file_name_by_candidate = {}


def index_resumes(resumes: Dict[str, str]) -> VectorStoreIndex:
    """
    Build an in-memory ChromaDB-backed VectorStoreIndex from resume texts.

    Args:
        resumes: mapping of {file_name: raw_resume_text}

    Returns:
        The constructed VectorStoreIndex.
    """
    global _chroma_client, _index, _file_name_by_candidate

    _get_embed_model()

    _chroma_client = chromadb.EphemeralClient()
    collection = _chroma_client.get_or_create_collection("resumes")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    documents: List[Document] = []
    for file_name, text in resumes.items():
        _file_name_by_candidate[file_name] = file_name
        documents.append(
            Document(
                text=text,
                metadata={"file_name": file_name},
                excluded_llm_metadata_keys=["file_name"],
            )
        )

    _index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=_embed_model,
    )
    logger.info("Indexed %d resumes into ChromaDB.", len(documents))
    return _index


def get_query_engine(similarity_top_k: int = 3) -> BaseQueryEngine:
    """
    Return a retrieval-only query engine over the currently built index.

    Uses response_mode="no_text" equivalent behaviour by relying on the
    retriever directly, since we don't want an LLM synthesizing an answer
    here (we only need raw retrieved chunks for the agents to reason over).
    """
    global _query_engine, _index

    if _index is None:
        raise RuntimeError("No resumes have been indexed yet. Call index_resumes() first.")

    if _query_engine is None:
        retriever = _index.as_retriever(similarity_top_k=similarity_top_k)
        _query_engine = retriever  # retriever exposes .retrieve(query)
    return _query_engine


def _format_nodes(nodes) -> str:
    """Format retrieved nodes into a readable text block for the LLM agent."""
    if not nodes:
        return "No relevant resume content found."

    chunks = []
    for node_with_score in nodes:
        node = node_with_score.node
        file_name = node.metadata.get("file_name", "unknown")
        score = getattr(node_with_score, "score", None)
        score_str = f" (relevance: {score:.2f})" if score is not None else ""
        chunks.append(f"--- Resume: {file_name}{score_str} ---\n{node.get_content()}")
    return "\n\n".join(chunks)


@tool("Resume Retrieval Tool")
def resume_retrieval_tool(query: str) -> str:
    """
    Search the indexed resumes for content relevant to the given query.
    Use this tool to look up specific skills, experience, education,
    certifications, or projects mentioned across candidate resumes.

    Args:
        query: A natural language search query, e.g. "Python experience"
               or "AWS certifications".

    Returns:
        A formatted string containing the most relevant resume excerpts.
    """
    try:
        engine = get_query_engine()
        nodes = engine.retrieve(query)
        return _format_nodes(nodes)
    except RuntimeError as exc:
        return f"Resume retrieval unavailable: {exc}"
    except Exception as exc:  # noqa: BLE001
        logger.exception("Resume retrieval tool failed")
        return f"Resume retrieval failed due to an internal error: {exc}"