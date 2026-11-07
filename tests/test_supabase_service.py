"""Unit tests for Supabase database service and Google authentication."""

import os
from unittest.mock import MagicMock, patch
import pytest

from langchain_app.supabase_service import (
    is_supabase_configured,
    get_supabase_client,
    upsert_google_user,
    save_chat_thread,
    save_document_record,
    list_user_documents,
    list_cloud_documents,
)
from langchain_app.auth_service import (
    is_google_auth_configured,
    get_google_auth_url,
    get_mock_google_user,
)


def test_is_supabase_configured_false(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    assert is_supabase_configured() is False


def test_is_supabase_configured_placeholder(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://tu_supabase_url.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "tu_supabase_key")
    assert is_supabase_configured() is False


def test_is_supabase_configured_true(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://xyzproject.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "sb_secret_token_12345")
    assert is_supabase_configured() is True


def test_upsert_google_user_offline_fallback(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    with patch("langchain_app.supabase_service.get_supabase_client", return_value=None):
        res = upsert_google_user(
            google_id="google_123",
            email="test@gmail.com",
            full_name="Tester Name",
        )
        assert res["google_id"] == "google_123"
        assert res["email"] == "test@gmail.com"
        assert "mock-" in res["id"]


def test_upsert_google_user_with_mocked_supabase():
    mock_client = MagicMock()
    mock_table = MagicMock()
    mock_upsert = MagicMock()
    mock_execute = MagicMock()

    mock_client.table.return_value = mock_table
    mock_table.upsert.return_value = mock_upsert
    mock_upsert.execute.return_value = MagicMock(data=[{
        "id": "uuid-1234-5678",
        "google_id": "google_999",
        "email": "user@gmail.com",
        "full_name": "Test User",
    }])

    with patch("langchain_app.supabase_service.get_supabase_client", return_value=mock_client):
        res = upsert_google_user(
            google_id="google_999",
            email="user@gmail.com",
            full_name="Test User",
        )
        assert res["id"] == "uuid-1234-5678"
        assert res["google_id"] == "google_999"


def test_save_chat_thread_mocked():
    mock_client = MagicMock()
    mock_table = MagicMock()
    mock_upsert = MagicMock()
    mock_client.table.return_value = mock_table
    mock_table.upsert.return_value = mock_upsert
    mock_upsert.execute.return_value = MagicMock(data=[{
        "id": "thread-abc",
        "title": "Chat sesión 1",
    }])

    with patch("langchain_app.supabase_service.get_supabase_client", return_value=mock_client):
        res = save_chat_thread(thread_id="thread-abc", title="Chat sesión 1")
        assert res["id"] == "thread-abc"


def test_save_document_record_mocked():
    mock_client = MagicMock()
    mock_table = MagicMock()
    mock_insert = MagicMock()
    mock_client.table.return_value = mock_table
    mock_table.insert.return_value = mock_insert
    mock_insert.execute.return_value = MagicMock(data=[{
        "id": "doc-uuid-1",
        "cloudinary_public_id": "langchain_app_pdfs/sample_123",
        "secure_url": "https://res.cloudinary.com/test.pdf",
        "file_name": "sample.pdf",
        "file_size_bytes": 1024,
    }])

    with patch("langchain_app.supabase_service.get_supabase_client", return_value=mock_client):
        res = save_document_record(
            cloudinary_public_id="langchain_app_pdfs/sample_123",
            secure_url="https://res.cloudinary.com/test.pdf",
            file_name="sample.pdf",
            file_size_bytes=1024,
        )
        assert res["cloudinary_public_id"] == "langchain_app_pdfs/sample_123"
        assert res["file_name"] == "sample.pdf"


def test_list_user_documents_and_tool():
    mock_client = MagicMock()
    mock_table = MagicMock()
    mock_select = MagicMock()
    mock_order = MagicMock()
    mock_limit = MagicMock()

    mock_client.table.return_value = mock_table
    mock_table.select.return_value = mock_select
    mock_select.order.return_value = mock_order
    mock_order.limit.return_value = mock_limit
    mock_limit.execute.return_value = MagicMock(data=[
        {
            "file_name": "informe_q1.pdf",
            "secure_url": "https://res.cloudinary.com/demo.pdf",
            "cloudinary_public_id": "pdfs/q1",
            "file_size_bytes": 5000,
        }
    ])

    with patch("langchain_app.supabase_service.get_supabase_client", return_value=mock_client):
        docs = list_user_documents()
        assert len(docs) == 1
        assert docs[0]["file_name"] == "informe_q1.pdf"

        tool_result = list_cloud_documents.invoke({"user_email": ""})
        assert "informe_q1.pdf" in tool_result
        assert "https://res.cloudinary.com/demo.pdf" in tool_result


def test_google_auth_helpers(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "12345.apps.googleusercontent.com")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "secret_abc_123")
    assert is_google_auth_configured() is True

    url = get_google_auth_url()
    assert "https://accounts.google.com/o/oauth2/v2/auth" in url
    assert "client_id=12345.apps.googleusercontent.com" in url

    # Test mock user
    user = get_mock_google_user(email="test.user@example.com", name="Test User")
    assert user["authenticated"] is True
    assert user["email"] == "test.user@example.com"


def test_calculate_md5():
    from langchain_app.supabase_service import calculate_md5
    md5_bytes = calculate_md5(b"sample data")
    md5_str = calculate_md5("sample data")
    assert len(md5_bytes) == 32
    assert md5_bytes == md5_str


def test_save_chat_message_mocked():
    from langchain_app.supabase_service import save_chat_message
    mock_client = MagicMock()
    mock_table = MagicMock()
    mock_insert = MagicMock()
    mock_client.table.return_value = mock_table
    mock_table.insert.return_value = mock_insert
    mock_insert.execute.return_value = MagicMock(data=[{
        "id": "msg-123",
        "thread_id": "thread-1",
        "sender": "user",
        "content": "¿Cuál es el resultado de 2 + 2?",
    }])

    with patch("langchain_app.supabase_service.get_supabase_client", return_value=mock_client), \
         patch("langchain_app.supabase_service.save_chat_thread", return_value={"id": "thread-1"}):
        res = save_chat_message(
            thread_id="thread-1",
            sender="user",
            content="¿Cuál es el resultado de 2 + 2?",
        )
        assert res["sender"] == "user"
        assert res["id"] == "msg-123"


def test_save_document_chunk_mocked():
    from langchain_app.supabase_service import save_document_chunk
    mock_client = MagicMock()
    mock_table = MagicMock()
    mock_insert = MagicMock()
    mock_client.table.return_value = mock_table
    mock_table.insert.return_value = mock_insert
    mock_insert.execute.return_value = MagicMock(data=[{
        "id": "chunk-1",
        "document_id": "doc-uuid-1",
        "chunk_index": 0,
        "content": "Texto del chunk",
    }])

    with patch("langchain_app.supabase_service.get_supabase_client", return_value=mock_client):
        res = save_document_chunk(
            document_id="doc-uuid-1",
            chunk_index=0,
            content="Texto del chunk",
            page_number=1,
            embedding=[0.1, 0.2, 0.3],
        )
        assert res["chunk_index"] == 0
        assert res["id"] == "chunk-1"


