"""Supabase database service conforming to the multi-level agent architecture.

Persists data across the 5 architectural levels defined in the specification:
1. Identidad & Auth (users): google_id (sub), email, full_name, avatar_url, last_login_at, created_at.
2. Conversaciones & Sesiones (chat_threads): thread_id (LangGraph), title, context_summary, timestamps.
3. Mensajes & Tool Calls (chat_messages): sender (user/assistant/tool), content, tool_name, tool_input, tool_output, tokens_consumed.
4. Archivos & Documentos (document_files): secure_url, cloudinary_public_id, file_name, mime_type, file_size_bytes, md5_checksum, processing_status.
5. Vectores & Embeddings RAG (document_chunks): document_id, chunk_index, content, page_number, embedding VECTOR(768).
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Optional
from dotenv import load_dotenv
from langchain_core.tools import tool
from langsmith import traceable

# Load environment variables
load_dotenv()

logger = logging.getLogger("langchain_app.supabase_service")

# Global client cache
_supabase_client = None

# Complete 5-Level SQL DDL for Supabase PostgreSQL + pgvector
SUPABASE_SCHEMA_SQL = """
-- Habilitar extensión vectorial para RAG
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Identidad & Auth
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    google_id VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    full_name VARCHAR(255),
    avatar_url TEXT,
    session_token TEXT,
    last_login_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Conversaciones & Sesiones (alineados al thread_id de LangGraph)
CREATE TABLE IF NOT EXISTS chat_threads (
    id VARCHAR(255) PRIMARY KEY, -- thread_id de LangGraph
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255) DEFAULT 'Nueva conversación',
    context_summary TEXT, -- Resumen acumulado para ventana deslizante
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Mensajes & Tool Calls (Trazabilidad exacta de ejecuciones)
CREATE TABLE IF NOT EXISTS chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id VARCHAR(255) REFERENCES chat_threads(id) ON DELETE CASCADE,
    sender VARCHAR(50) NOT NULL, -- 'user', 'assistant', 'tool', 'system'
    content TEXT NOT NULL,
    tool_name VARCHAR(100),
    tool_input JSONB,
    tool_output TEXT,
    tokens_consumed INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 4. Archivos & Documentos respaldados en Cloudinary
CREATE TABLE IF NOT EXISTS document_files (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id VARCHAR(255) REFERENCES chat_threads(id) ON DELETE SET NULL,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    cloudinary_public_id VARCHAR(255) NOT NULL,
    secure_url TEXT NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    mime_type VARCHAR(100) DEFAULT 'application/pdf',
    file_size_bytes INT,
    md5_checksum VARCHAR(64),
    processing_status VARCHAR(50) DEFAULT 'completed', -- 'uploaded', 'processing', 'completed', 'error'
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 5. Chunks de PDFs para búsqueda semántica (RAG con pgvector)
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES document_files(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    page_number INT,
    embedding VECTOR(768), -- Dimensión estándar de embeddings (Gemini text-embedding-004)
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Índices para optimización de consultas
CREATE INDEX IF NOT EXISTS idx_chat_messages_thread ON chat_messages(thread_id);
CREATE INDEX IF NOT EXISTS idx_document_files_user ON document_files(user_id);
"""


def is_supabase_configured() -> bool:
    """Check if valid Supabase connection credentials exist in the environment."""
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_KEY", "").strip()
    placeholder_terms = {"tu_supabase_url", "tu_supabase_key", "your_supabase_url", "https://your-project.supabase.co"}
    if not url or not key:
        return False
    if any(p in url.lower() or p in key.lower() for p in placeholder_terms):
        return False
    return True


def get_supabase_client():
    """Create or return cached Supabase Client instance."""
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    if not is_supabase_configured():
        return None

    try:
        from supabase import create_client
        url = os.getenv("SUPABASE_URL", "").strip()
        key = os.getenv("SUPABASE_KEY", "").strip()
        _supabase_client = create_client(url, key)
        return _supabase_client
    except Exception as exc:
        logger.error("Failed to initialize Supabase client: %s", exc)
        return None


def calculate_md5(data: bytes | str) -> str:
    """Calculate MD5 checksum for binary or text content."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.md5(data).hexdigest()


# -----------------------------------------------------------------------------
# Level 1: Identidad & Auth
# -----------------------------------------------------------------------------
@traceable(run_type="tool", name="supabase_upsert_user", tags=["supabase", "auth"])
def upsert_google_user(
    google_id: str,
    email: str,
    full_name: Optional[str] = None,
    avatar_url: Optional[str] = None,
    session_token: Optional[str] = None,
) -> dict[str, Any]:
    """Sync or create a user in Supabase from Google OAuth profile data."""
    client = get_supabase_client()
    if not client:
        return {
            "id": f"mock-{google_id[:8]}",
            "google_id": google_id,
            "email": email,
            "full_name": full_name or email.split("@")[0],
            "avatar_url": avatar_url,
            "session_token": session_token,
            "status": "local_mock_unconfigured_supabase",
        }

    try:
        data = {
            "google_id": google_id,
            "email": email,
            "full_name": full_name or "",
            "avatar_url": avatar_url or "",
            "session_token": session_token or "",
        }
        res = client.table("users").upsert(data, on_conflict="google_id").execute()
        if res.data and len(res.data) > 0:
            return res.data[0]
        return data
    except Exception as exc:
        logger.error("Error upserting user to Supabase: %s", exc)
        return {
            "google_id": google_id,
            "email": email,
            "full_name": full_name,
            "avatar_url": avatar_url,
            "error": str(exc),
        }


# -----------------------------------------------------------------------------
# Level 2: Conversaciones & Sesiones
# -----------------------------------------------------------------------------
@traceable(run_type="tool", name="supabase_save_thread", tags=["supabase", "chat"])
def save_chat_thread(
    thread_id: str,
    user_id: Optional[str] = None,
    title: str = "Nueva conversación",
    context_summary: Optional[str] = None,
) -> dict[str, Any]:
    """Record or update a chat thread session with context summary in Supabase."""
    client = get_supabase_client()
    if not client:
        return {
            "id": thread_id,
            "user_id": user_id,
            "title": title,
            "context_summary": context_summary,
            "status": "offline",
        }

    try:
        data: dict[str, Any] = {
            "id": thread_id,
            "title": title,
        }
        if context_summary:
            data["context_summary"] = context_summary
        if user_id and not user_id.startswith("mock-"):
            data["user_id"] = user_id

        res = client.table("chat_threads").upsert(data, on_conflict="id").execute()
        if res.data:
            return res.data[0]
        return data
    except Exception as exc:
        logger.warning("Could not persist thread to Supabase: %s", exc)
        return {"id": thread_id, "title": title, "error": str(exc)}


@traceable(run_type="tool", name="supabase_get_user_threads", tags=["supabase", "chat"])
def get_user_chat_threads(user_id: Optional[str] = None, limit: int = 15) -> list[dict[str, Any]]:
    """Retrieve chat threads belonging to a user, ordered by most recently updated."""
    client = get_supabase_client()
    if not client:
        return []

    try:
        query = client.table("chat_threads").select("*")
        if user_id and not user_id.startswith("mock-"):
            query = query.eq("user_id", user_id)
        res = query.order("updated_at", desc=True).limit(limit).execute()
        return res.data or []
    except Exception as exc:
        logger.warning("Could not fetch user chat threads from Supabase: %s", exc)
        return []


@traceable(run_type="tool", name="supabase_get_thread_messages", tags=["supabase", "chat"])
def get_thread_messages(thread_id: str) -> list[dict[str, Any]]:
    """Retrieve all messages for a specific chat thread in chronological order."""
    client = get_supabase_client()
    if not client:
        return []

    try:
        res = client.table("chat_messages").select("*").eq("thread_id", thread_id).order("created_at", desc=False).execute()
        return res.data or []
    except Exception as exc:
        logger.warning("Could not fetch messages for thread '%s': %s", thread_id, exc)
        return []


@traceable(run_type="tool", name="supabase_delete_thread", tags=["supabase", "chat"])
def delete_chat_thread(thread_id: str) -> bool:
    """Delete a chat thread and all its messages."""
    client = get_supabase_client()
    if not client:
        return False

    try:
        client.table("chat_threads").delete().eq("id", thread_id).execute()
        return True
    except Exception as exc:
        logger.warning("Could not delete chat thread '%s': %s", thread_id, exc)
        return False


# -----------------------------------------------------------------------------
# Level 3: Mensajes & Tool Calls
# -----------------------------------------------------------------------------
@traceable(run_type="tool", name="supabase_save_message", tags=["supabase", "messages"])
def save_chat_message(
    thread_id: str,
    sender: str,
    content: str,
    tool_name: Optional[str] = None,
    tool_input: Optional[dict[str, Any]] = None,
    tool_output: Optional[str] = None,
    tokens_consumed: Optional[int] = None,
) -> dict[str, Any]:
    """Persist an individual message turn or tool invocation for exact auditability."""
    client = get_supabase_client()
    record = {
        "thread_id": thread_id,
        "sender": sender,
        "content": content,
        "tool_name": tool_name,
        "tool_input": tool_input,
        "tool_output": tool_output,
        "tokens_consumed": tokens_consumed,
    }

    if not client:
        record["status"] = "offline_local"
        return record

    try:
        # Ensure thread exists first
        save_chat_thread(thread_id=thread_id)
        res = client.table("chat_messages").insert(record).execute()
        if res.data:
            return res.data[0]
        return record
    except Exception as exc:
        logger.warning("Could not save chat message to Supabase: %s", exc)
        record["error"] = str(exc)
        return record


# -----------------------------------------------------------------------------
# Level 4: Archivos & Documentos
# -----------------------------------------------------------------------------
@traceable(run_type="tool", name="supabase_save_document", tags=["supabase", "documents"])
def save_document_record(
    cloudinary_public_id: str,
    secure_url: str,
    file_name: str,
    file_size_bytes: int,
    user_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    mime_type: str = "application/pdf",
    md5_checksum: Optional[str] = None,
    processing_status: str = "completed",
) -> dict[str, Any]:
    """Save metadata of a Cloudinary-uploaded PDF into Supabase."""
    client = get_supabase_client()
    record = {
        "cloudinary_public_id": cloudinary_public_id,
        "secure_url": secure_url,
        "file_name": file_name,
        "file_size_bytes": file_size_bytes,
        "mime_type": mime_type,
        "md5_checksum": md5_checksum,
        "processing_status": processing_status,
        "user_id": user_id if (user_id and not user_id.startswith("mock-")) else None,
        "thread_id": thread_id,
    }

    if not client:
        record["status"] = "offline_local"
        return record

    try:
        if thread_id:
            save_chat_thread(thread_id=thread_id, user_id=user_id)
        res = client.table("document_files").insert(record).execute()
        if res.data:
            return res.data[0]
        return record
    except Exception as exc:
        logger.warning("Could not persist document metadata to Supabase: %s", exc)
        record["error"] = str(exc)
        return record


@traceable(run_type="tool", name="supabase_list_documents", tags=["supabase", "documents"])
def list_user_documents(user_id: Optional[str] = None, limit: int = 10) -> list[dict[str, Any]]:
    """Retrieve document files uploaded by a user or thread."""
    client = get_supabase_client()
    if not client:
        return []

    try:
        query = client.table("document_files").select("*").order("created_at", desc=True).limit(limit)
        if user_id and not user_id.startswith("mock-"):
            query = query.eq("user_id", user_id)
        res = query.execute()
        return res.data or []
    except Exception as exc:
        logger.warning("Could not fetch user documents from Supabase: %s", exc)
        return []


# -----------------------------------------------------------------------------
# Level 5: Vectores & Chunks RAG
# -----------------------------------------------------------------------------
@traceable(run_type="tool", name="supabase_save_chunk", tags=["supabase", "rag"])
def save_document_chunk(
    document_id: str,
    chunk_index: int,
    content: str,
    page_number: Optional[int] = None,
    embedding: Optional[list[float]] = None,
) -> dict[str, Any]:
    """Store text chunk and vector embedding for semantic search in pgvector."""
    client = get_supabase_client()
    record = {
        "document_id": document_id,
        "chunk_index": chunk_index,
        "content": content,
        "page_number": page_number,
    }
    if embedding:
        record["embedding"] = embedding

    if not client:
        record["status"] = "offline_local"
        return record

    try:
        res = client.table("document_chunks").insert(record).execute()
        if res.data:
            return res.data[0]
        return record
    except Exception as exc:
        logger.warning("Could not save chunk to Supabase: %s", exc)
        record["error"] = str(exc)
        return record


@tool
def list_cloud_documents(user_email: str = "") -> str:
    """Retrieve the list of documents uploaded to Cloudinary and registered in the database.
    
    Args:
        user_email: Optional email to filter user documents.
    """
    docs = list_user_documents(limit=10)
    if not docs:
        return "No hay documentos registrados en la base de datos de Supabase."
    
    lines = [f"📄 **Documentos encontrados ({len(docs)}):**"]
    for d in docs:
        md5_info = f" | MD5: `{d.get('md5_checksum')[:8]}...`" if d.get('md5_checksum') else ""
        lines.append(
            f"- **{d.get('file_name', 'Archivo')}**: [Ver en Cloudinary]({d.get('secure_url')}) "
            f"(ID: `{d.get('cloudinary_public_id')}`, Tamaño: {d.get('file_size_bytes', 0)} bytes{md5_info})"
        )
    return "\n".join(lines)
