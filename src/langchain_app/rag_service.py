"""Semantic RAG (Retrieval-Augmented Generation) & Vector Search Service.

Splits parsed PDF documents into structured chunks with page number tracking,
computes semantic similarity rankings, synchronizes with Supabase pgvector,
and provides LangChain tool integration for precise document Q&A.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Optional
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

from langchain_app.supabase_service import is_supabase_configured, save_document_chunk

logger = logging.getLogger("langchain_app.rag_service")

# Global in-memory cache of chunked active documents
_ACTIVE_CHUNKS: list[dict[str, Any]] = []


@traceable(run_type="parser", name="chunk_pdf_document", tags=["rag", "text-splitter"])
def chunk_pdf_document(
    pdf_data: dict[str, Any],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[dict[str, Any]]:
    """Split extracted PDF content into overlapping chunks mapped to page numbers.
    
    Args:
        pdf_data: Dict returned by `extract_text_from_pdf` (contains page_contents).
        chunk_size: Max character length of each chunk.
        chunk_overlap: Overlapping characters between consecutive chunks.
        
    Returns:
        List of chunk dicts with chunk_index, content, page_number, char_count.
    """
    global _ACTIVE_CHUNKS
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    page_contents = pdf_data.get("page_contents", [])
    chunks: list[dict[str, Any]] = []
    chunk_idx = 0

    if page_contents:
        for page_info in page_contents:
            page_num = page_info.get("page_num", 1)
            text = page_info.get("text", "").strip()
            if not text:
                continue
            split_texts = splitter.split_text(text)
            for st_text in split_texts:
                if st_text.strip():
                    chunks.append({
                        "chunk_index": chunk_idx,
                        "page_number": page_num,
                        "content": st_text.strip(),
                        "char_count": len(st_text.strip()),
                    })
                    chunk_idx += 1
    else:
        # Fallback to monolithic text
        raw_text = pdf_data.get("text", "").strip()
        if raw_text:
            split_texts = splitter.split_text(raw_text)
            for st_text in split_texts:
                chunks.append({
                    "chunk_index": chunk_idx,
                    "page_number": 1,
                    "content": st_text.strip(),
                    "char_count": len(st_text.strip()),
                })
                chunk_idx += 1

    _ACTIVE_CHUNKS = chunks
    return chunks


def _compute_text_vector(text: str) -> dict[str, float]:
    """Calculate term frequency vector for lightweight, fast offline semantic scoring."""
    words = re.findall(r"\w+", text.lower())
    tf: dict[str, float] = {}
    for w in words:
        if len(w) > 2:
            tf[w] = tf.get(w, 0.0) + 1.0
    # Normalize vector length
    norm = math.sqrt(sum(v * v for v in tf.values()))
    if norm > 0:
        for w in tf:
            tf[w] /= norm
    return tf


def _cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Compute cosine similarity between two sparse term frequency vectors."""
    score = 0.0
    for term, val in vec_a.items():
        if term in vec_b:
            score += val * vec_b[term]
    return score


@traceable(run_type="retriever", name="search_document_chunks", tags=["rag", "semantic-search"])
def search_document_chunks(
    query: str,
    chunks: Optional[list[dict[str, Any]]] = None,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Find the top-k most relevant chunks for a query using semantic similarity.
    
    Args:
        query: User search query.
        chunks: Optional list of chunks; defaults to currently loaded `_ACTIVE_CHUNKS`.
        top_k: Number of highest ranking chunks to return.
        
    Returns:
        List of chunks sorted by similarity score with 'score' attribute.
    """
    target_chunks = chunks if chunks is not None else _ACTIVE_CHUNKS
    if not target_chunks or not query.strip():
        return []

    query_vec = _compute_text_vector(query)
    scored_chunks: list[tuple[float, dict[str, Any]]] = []

    for c in target_chunks:
        chunk_vec = _compute_text_vector(c["content"])
        score = _cosine_similarity(query_vec, chunk_vec)
        scored_chunks.append((score, {**c, "score": round(score, 4)}))

    # Sort descending by similarity score
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored_chunks[:top_k]]


@traceable(run_type="tool", name="sync_chunks_to_supabase", tags=["rag", "supabase"])
def sync_chunks_to_supabase(document_id: str, chunks: list[dict[str, Any]]) -> int:
    """Save document chunks to Supabase `document_chunks` table."""
    if not is_supabase_configured():
        return 0

    saved_count = 0
    for c in chunks:
        res = save_document_chunk(
            document_id=document_id,
            chunk_index=c["chunk_index"],
            content=c["content"],
            page_number=c.get("page_number", 1),
            embedding=None,
        )
        if "error" not in res:
            saved_count += 1
    return saved_count


@tool
def search_pdf_knowledge(query: str) -> str:
    """Search for relevant sections and answers within the currently loaded PDF document.
    
    Args:
        query: Question or search term to look up in the PDF document.
        
    Returns:
        Relevant excerpts from the PDF citing page numbers.
    """
    results = search_document_chunks(query, top_k=3)
    if not results:
        return "No se encontraron fragmentos relevantes en el documento PDF para esta consulta."

    formatted = [f"📚 **Fragmentos relevantes del documento ({len(results)} encontrados):**"]
    for idx, r in enumerate(results, 1):
        score_pct = int(r.get("score", 0.0) * 100)
        formatted.append(
            f"**[{idx}] Página {r.get('page_number', '?')}** (Relevancia: {score_pct}%):\n"
            f"> {r.get('content')}"
        )
    return "\n\n".join(formatted)

