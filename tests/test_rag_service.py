"""Unit tests for Semantic RAG, text chunking, and vector search."""

import pytest
from langchain_app.rag_service import (
    chunk_pdf_document,
    search_document_chunks,
    sync_chunks_to_supabase,
    search_pdf_knowledge,
)


def test_chunk_pdf_document_multi_page():
    sample_pdf_data = {
        "filename": "quarterly_earnings.pdf",
        "total_pages": 2,
        "page_contents": [
            {
                "page_num": 1,
                "text": "Quarterly Earnings Report. The company achieved 25 percent revenue growth in Q3. Operating expenses were kept low.",
            },
            {
                "page_num": 2,
                "text": "Future Projections. In Q4 we expect to expand AI agent infrastructure and launch new cloud features with LangGraph.",
            },
        ],
    }

    chunks = chunk_pdf_document(sample_pdf_data, chunk_size=80, chunk_overlap=20)
    assert len(chunks) >= 2
    assert chunks[0]["page_number"] == 1
    assert "chunk_index" in chunks[0]
    assert "content" in chunks[0]


def test_chunk_pdf_document_monolithic_text_fallback():
    sample_data = {
        "filename": "single_text.pdf",
        "text": "This is a monolithic fallback text document without page breakdown.",
    }
    chunks = chunk_pdf_document(sample_data, chunk_size=40, chunk_overlap=10)
    assert len(chunks) >= 1
    assert chunks[0]["page_number"] == 1


def test_search_document_chunks_semantic_ranking():
    chunks = [
        {
            "chunk_index": 0,
            "page_number": 1,
            "content": "Deep learning models require neural network architectures and high performance GPUs.",
        },
        {
            "chunk_index": 1,
            "page_number": 2,
            "content": "Vegetable farming requires fertile soil, irrigation systems, and proper seasonal sunlight.",
        },
        {
            "chunk_index": 2,
            "page_number": 3,
            "content": "Artificial intelligence algorithms process data through neural layers and tensors.",
        },
    ]

    results = search_document_chunks("neural network artificial intelligence", chunks=chunks, top_k=2)
    assert len(results) == 2
    # The top result should be AI/neural network related, not farming
    assert "neural" in results[0]["content"].lower() or "artificial" in results[0]["content"].lower()
    assert "farming" not in results[0]["content"].lower()


def test_search_document_chunks_empty():
    assert search_document_chunks("", chunks=[]) == []
    assert search_document_chunks("test query", chunks=[]) == []


def test_search_pdf_knowledge_tool():
    # Load chunks into active state
    sample_data = {
        "page_contents": [
            {"page_num": 1, "text": "Cloudinary stores image and PDF assets securely with CDN acceleration."},
            {"page_num": 2, "text": "Supabase provides PostgreSQL databases with pgvector embeddings."},
        ]
    }
    chunk_pdf_document(sample_data, chunk_size=200, chunk_overlap=20)

    tool_res = search_pdf_knowledge.invoke({"query": "How does Cloudinary store assets?"})
    assert "Cloudinary" in tool_res
    assert "Página 1" in tool_res


def test_sync_chunks_to_supabase_unconfigured(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    count = sync_chunks_to_supabase("doc-123", [{"chunk_index": 0, "content": "test"}])
    assert count == 0

