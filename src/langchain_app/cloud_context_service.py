"""Cloudinary Document Knowledge & Context Service for LangGraph.

Detects allusions to files uploaded to Cloudinary/Supabase in user queries,
ensures their parsed chunks are synchronized in Supabase, and generates
indispensable grounding context blocks for LangGraph multi-tool reasoning.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Optional

import requests
from langchain_core.tools import tool
from langsmith import traceable

from langchain_app.pdf_service import extract_text_from_pdf
from langchain_app.rag_service import chunk_pdf_document, search_document_chunks, sync_chunks_to_supabase
from langchain_app.supabase_service import (
    get_document_by_name,
    get_document_chunks_by_document_id,
    is_supabase_configured,
    list_user_documents,
)

logger = logging.getLogger("langchain_app.cloud_context_service")

# In-memory chunk cache to avoid repeated network lookups
_DOC_CHUNKS_CACHE: dict[str, list[dict[str, Any]]] = {}

# Terms indicating allusion to uploaded/cloud documents
DOCUMENT_ALLUSION_KEYWORDS = {
    "fichero", "ficheros", "archivo", "archivos", "documento", "documentos",
    "pdf", "cloudinary", "nube", "subido", "subidos", "cargado", "cargados",
    "adjunto", "adjuntos", "examen", "examenes", "analisis", "resultado",
    "resultados", "estudio", "estudios", "laboratorio", "sangre", "informe",
    "informes", "reporte", "reportes", "hemoglobina", "glucosa", "plaquetas",
    "colesterol", "trigliceridos", "orina", "heces", "clinico", "medico",
}


def normalize_term(text: str) -> str:
    """Normalize text removing accents and non-alphanumeric characters."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    ascii_text = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"[^\w\s]", " ", ascii_text).strip()


def detect_cloud_document_allusion(
    query: str,
    user_documents: Optional[list[dict[str, Any]]] = None,
) -> tuple[bool, Optional[dict[str, Any]]]:
    """Detect if the query refers to uploaded Cloudinary/Supabase files.
    
    Returns:
        (has_allusion, matching_document_record_or_none)
    """
    clean_query = normalize_term(query)
    words = set(clean_query.split())

    # 1. Match against known document filenames
    docs = user_documents if user_documents is not None else list_user_documents(limit=10)
    for doc in docs:
        raw_name = doc.get("file_name", "")
        stem = normalize_term(raw_name.replace(".pdf", ""))
        # If document name or parts of it appear in query
        if stem and (stem in clean_query or any(part in clean_query for part in stem.split() if len(part) >= 4)):
            return True, doc

    # 2. Match against document allusion keywords
    matches = words.intersection(DOCUMENT_ALLUSION_KEYWORDS)
    if matches and docs:
        # Match found and user has uploaded documents: return most recent doc as candidate
        return True, docs[0]

    return False, None


@traceable(run_type="parser", name="sync_cloud_doc_chunks", tags=["cloudinary", "rag", "chunks"])
def ensure_document_synced_and_chunked(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Ensure document chunks exist in memory and in Supabase `document_chunks` table."""
    doc_id = doc.get("id")
    if not doc_id:
        return []

    # Check cache first
    if doc_id in _DOC_CHUNKS_CACHE and _DOC_CHUNKS_CACHE[doc_id]:
        return _DOC_CHUNKS_CACHE[doc_id]

    # Check Supabase database
    existing_chunks = get_document_chunks_by_document_id(doc_id)
    if existing_chunks:
        _DOC_CHUNKS_CACHE[doc_id] = existing_chunks
        return existing_chunks

    # If no chunks in Supabase, download from Cloudinary URL and parse
    secure_url = doc.get("secure_url")
    if not secure_url:
        logger.warning("Document %s has no secure_url for chunking.", doc_id)
        return []

    try:
        logger.info("Downloading '%s' from Cloudinary to generate chunks...", doc.get("file_name"))
        resp = requests.get(secure_url, timeout=20)
        if resp.status_code != 200:
            logger.error("Failed to download PDF from Cloudinary (HTTP %s)", resp.status_code)
            return []

        pdf_info = extract_text_from_pdf(resp.content)
        chunks = chunk_pdf_document(pdf_info)

        if is_supabase_configured() and chunks:
            sync_chunks_to_supabase(doc_id, chunks)

        _DOC_CHUNKS_CACHE[doc_id] = chunks
        return chunks
    except Exception as exc:
        logger.error("Error syncing document chunks from Cloudinary: %s", exc)
        return []


@traceable(run_type="retriever", name="get_cloud_documents_context", tags=["cloudinary", "rag", "context"])
def get_cloud_documents_context(
    query: str,
    user_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    top_k: int = 6,
) -> Optional[str]:
    """Build an authoritative context prompt when the query references uploaded Cloudinary files.
    
    Returns:
        Formatted context string with extracted chunks and critical response instructions,
        or None if no allusion or documents were found.
    """
    user_docs = list_user_documents(user_id=user_id, limit=10)
    if not user_docs:
        return None

    has_allusion, matched_doc = detect_cloud_document_allusion(query, user_documents=user_docs)
    if not has_allusion or not matched_doc:
        return None

    chunks = ensure_document_synced_and_chunked(matched_doc)
    if not chunks:
        return None

    # Retrieve most relevant chunks for the user's specific question
    relevant_chunks = search_document_chunks(query, chunks=chunks, top_k=top_k)
    if not relevant_chunks:
        relevant_chunks = chunks[:top_k]

    file_name = matched_doc.get("file_name", "documento.pdf")
    secure_url = matched_doc.get("secure_url", "#")
    total_pages = max([c.get("page_number", 1) for c in chunks], default=1)

    chunk_blocks = []
    for c in relevant_chunks:
        p_num = c.get("page_number", 1)
        c_text = c.get("content", "").strip()
        chunk_blocks.append(f"[Página {p_num}]\n{c_text}")

    evidence_text = "\n\n---\n\n".join(chunk_blocks)

    context_prompt = (
        "======================================================================\n"
        "📁 CONTEXTO DE CONOCIMIENTO INDISPENSABLE: FICHEROS SUBIDOS EN CLOUDINARY\n"
        "======================================================================\n"
        f"Fichero: {file_name}\n"
        f"Almacenamiento Seguro Cloudinary: {secure_url}\n"
        f"Total de Páginas: {total_pages} | Chunks Relevantes: {len(relevant_chunks)}\n\n"
        "EVIDENCIA EXTRAÍDA DIRECTAMENTE DEL DOCUMENTO:\n"
        "----------------------------------------------------------------------\n"
        f"{evidence_text}\n"
        "----------------------------------------------------------------------\n\n"
        "DIRECTIVA CRÍTICA DE RESPUESTA PARA EL MODELO:\n"
        "1. El usuario está haciendo alusión directa a su fichero subido en Cloudinary.\n"
        "2. Este documento adjunto es tu FUENTE PRINCIPAL E INDISPENSABLE DE VERDAD.\n"
        "3. Responde a la consulta basándote estricta y detalladamente en los datos, "
        "valores, tablas y referencias extraídos de este fichero.\n"
        f"4. Cita explícitamente el nombre del fichero ('{file_name}') y los números de página "
        "correspondientes en tu conclusión.\n"
        "======================================================================\n"
    )

    return context_prompt


@tool
def read_cloud_document(query: str, document_name: str = "") -> str:
    """Read and extract specific knowledge from documents/PDFs uploaded by the user to Cloudinary.
    
    Use this tool whenever the user asks about their uploaded files, blood tests, medical reports,
    receipts, invoices, or any PDF documents backed up in Cloudinary.
    
    Args:
        query: Specific question or information to look up inside the document.
        document_name: Optional name of the document (e.g. 'resultados.pdf').
    """
    print(f"  🛠️  [EXECUTION: LangChain @tool 'read_cloud_document'] Query='{query}', DocName='{document_name}'")
    context = get_cloud_documents_context(query)
    if not context:
        # Fallback to looking up document by name
        if document_name:
            doc = get_document_by_name(document_name)
            if doc:
                chunks = ensure_document_synced_and_chunked(doc)
                if chunks:
                    top_chunks = search_document_chunks(query, chunks=chunks, top_k=5)
                    formatted = [f"[Página {c.get('page_number', 1)}] {c.get('content')}" for c in top_chunks]
                    return f"Contenido extraído de '{doc.get('file_name')}':\n\n" + "\n\n".join(formatted)

        return (
            "No se encontraron documentos coincidentes en Cloudinary o el documento no contiene datos para la consulta. "
            "Asegúrate de que el archivo PDF haya sido subido previamente."
        )

    return context
