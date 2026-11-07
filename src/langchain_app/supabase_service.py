"""Supabase database service for user profiles, chat threads, and document records.

Provides client instantiation, table schema definitions, and helpers for:
1. Google OAuth user provisioning (users table).
2. Chat thread synchronization for LangGraph sessions (chat_threads table).
3. Cloudinary PDF upload metadata tracking (document_files table).
4. Chunk storage for semantic search and RAG (document_chunks table).
"""

from __future__ import annotations

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

# SQL DDL for table creation (can be executed in Supabase SQL editor or automated)
SUPABASE_SCHEMA_SQL = """
-- 1. Usuarios vinculados con Google OAuth
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    google_id VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    full_name VARCHAR(255),
    avatar_url TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Hilos de conversación (alineados al thread_id de LangGraph)
CREATE TABLE IF NOT EXISTS chat_threads (
    id VARCHAR(255) PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255) DEFAULT 'Nueva conversación',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Archivos subidos a Cloudinary
CREATE TABLE IF NOT EXISTS document_files (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id VARCHAR(255) REFERENCES chat_threads(id) ON DELETE SET NULL,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    cloudinary_public_id VARCHAR(255) NOT NULL,
    secure_url TEXT NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    file_size_bytes INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 4. Chunks de PDFs para búsqueda semántica (RAG)
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES document_files(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    page_number INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
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
    """Create or return cached Supabase Client instance.
    
    Returns:
        Client object or None if unconfigured.
    """
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    if not is_supabase_configured():
        return None

    try:
        from supabase import create_client, Client
        url = os.getenv("SUPABASE_URL", "").strip()
        key = os.getenv("SUPABASE_KEY", "").strip()
        _supabase_client = create_client(url, key)
        return _supabase_client
    except Exception as exc:
        logger.error("Failed to initialize Supabase client: %s", exc)
        return None


@traceable(run_type="tool", name="supabase_upsert_user", tags=["supabase", "auth"])
def upsert_google_user(
    google_id: str,
    email: str,
    full_name: Optional[str] = None,
    avatar_url: Optional[str] = None,
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
            "status": "local_mock_unconfigured_supabase",
        }

    try:
        data = {
            "google_id": google_id,
            "email": email,
            "full_name": full_name or "",
            "avatar_url": avatar_url or "",
        }
        # Upsert based on google_id
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


@traceable(run_type="tool", name="supabase_save_thread", tags=["supabase", "chat"])
def save_chat_thread(
    thread_id: str,
    user_id: Optional[str] = None,
    title: str = "Nueva conversación",
) -> dict[str, Any]:
    """Record or update a chat thread session in Supabase."""
    client = get_supabase_client()
    if not client:
        return {"id": thread_id, "user_id": user_id, "title": title, "status": "offline"}

    try:
        data: dict[str, Any] = {
            "id": thread_id,
            "title": title,
        }
        if user_id and not user_id.startswith("mock-"):
            data["user_id"] = user_id

        res = client.table("chat_threads").upsert(data, on_conflict="id").execute()
        if res.data:
            return res.data[0]
        return data
    except Exception as exc:
        logger.warning("Could not persist thread to Supabase: %s", exc)
        return {"id": thread_id, "title": title, "error": str(exc)}


@traceable(run_type="tool", name="supabase_save_document", tags=["supabase", "documents"])
def save_document_record(
    cloudinary_public_id: str,
    secure_url: str,
    file_name: str,
    file_size_bytes: int,
    user_id: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> dict[str, Any]:
    """Save metadata of a Cloudinary-uploaded PDF into Supabase."""
    client = get_supabase_client()
    record = {
        "cloudinary_public_id": cloudinary_public_id,
        "secure_url": secure_url,
        "file_name": file_name,
        "file_size_bytes": file_size_bytes,
        "user_id": user_id if (user_id and not user_id.startswith("mock-")) else None,
        "thread_id": thread_id,
    }

    if not client:
        record["status"] = "offline_local"
        return record

    try:
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
        lines.append(
            f"- **{d.get('file_name', 'Archivo')}**: [Ver en Cloudinary]({d.get('secure_url')}) "
            f"(ID: `{d.get('cloudinary_public_id')}`, Tamaño: {d.get('file_size_bytes', 0)} bytes)"
        )
    return "\n".join(lines)
