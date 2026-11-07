"""Unit and integration tests for cloud_context_service module."""

from unittest.mock import MagicMock, patch
import pytest

from langchain_app.cloud_context_service import (
    detect_cloud_document_allusion,
    ensure_document_synced_and_chunked,
    get_cloud_documents_context,
    read_cloud_document,
)


def test_detect_cloud_document_allusion_keywords() -> None:
    sample_docs = [
        {"id": "doc-1", "file_name": "resultados.pdf", "secure_url": "https://example.com/res.pdf"}
    ]

    # Query with allusion to blood test and exams
    has_allusion, matched = detect_cloud_document_allusion(
        "Dame una conclusión de los resultados de mis exámenes de sangre",
        user_documents=sample_docs,
    )
    assert has_allusion is True
    assert matched is not None
    assert matched["id"] == "doc-1"

    # Query explicitly mentioning Cloudinary
    has_allusion2, _ = detect_cloud_document_allusion(
        "Qué archivos tengo subidos a cloudinary?",
        user_documents=sample_docs,
    )
    assert has_allusion2 is True

    # Pure conversational query without allusion
    has_allusion3, matched3 = detect_cloud_document_allusion(
        "Hola, cómo estás el día de hoy?",
        user_documents=sample_docs,
    )
    assert has_allusion3 is False
    assert matched3 is None


def test_detect_cloud_document_allusion_empty_docs() -> None:
    has_allusion, matched = detect_cloud_document_allusion(
        "Dame los resultados del archivo",
        user_documents=[],
    )
    assert has_allusion is False
    assert matched is None


@patch("langchain_app.cloud_context_service.get_document_chunks_by_document_id")
def test_ensure_document_synced_from_db(mock_get_chunks: MagicMock) -> None:
    mock_get_chunks.return_value = [
        {"chunk_index": 0, "page_number": 1, "content": "Hemoglobina: 14.5 g/dl"}
    ]
    doc = {"id": "doc-test-123", "file_name": "analisis.pdf", "secure_url": "https://example.com/a.pdf"}

    chunks = ensure_document_synced_and_chunked(doc)
    assert len(chunks) == 1
    assert "Hemoglobina" in chunks[0]["content"]


@patch("langchain_app.cloud_context_service.list_user_documents")
@patch("langchain_app.cloud_context_service.ensure_document_synced_and_chunked")
def test_get_cloud_documents_context(
    mock_ensure: MagicMock,
    mock_list_docs: MagicMock,
) -> None:
    mock_list_docs.return_value = [
        {"id": "doc-blood", "file_name": "resultados.pdf", "secure_url": "https://cloudinary.com/test"}
    ]
    mock_ensure.return_value = [
        {"chunk_index": 0, "page_number": 1, "content": "Página 1: Glucosa 95 mg/dl, Colesterol 180 mg/dl."},
        {"chunk_index": 1, "page_number": 2, "content": "Página 2: Plaquetas 250,000 /mm3."},
    ]

    context = get_cloud_documents_context("Cuál es mi nivel de glucosa en los resultados?")
    assert context is not None
    assert "FICHERO ADJUNTO" in context
    assert "resultados.pdf" in context
    assert "EVIDENCIA EXTRAÍDA DEL DOCUMENTO" in context


@patch("langchain_app.cloud_context_service.get_cloud_documents_context")
def test_read_cloud_document_tool(mock_get_ctx: MagicMock) -> None:
    mock_get_ctx.return_value = "Mocked document context for query"
    res = read_cloud_document.invoke({"query": "mis examenes"})
    assert "Mocked document context" in res

