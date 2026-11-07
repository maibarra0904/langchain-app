import io
from pathlib import Path
from typing import BinaryIO

from langchain_core.tools import tool
from langsmith import traceable
from pypdf import PdfReader


@traceable(run_type="retriever", name="pdf_text_extraction", tags=["pdf", "document-parser"])
def extract_text_from_pdf(
    pdf_source: bytes | str | Path | BinaryIO,
    max_pages: int = 50,
) -> dict:
    """
    Extract text, page count, and metadata from a PDF file (path or bytes).
    Returns a dict with:
      - filename: str
      - total_pages: int
      - pages_extracted: int
      - text: str (concatenated text)
      - page_contents: list[dict(page_num, text)]
      - metadata: dict
    """
    filename = "document.pdf"
    if isinstance(pdf_source, (str, Path)):
        path_obj = Path(pdf_source)
        if not path_obj.exists():
            raise FileNotFoundError(f"PDF file not found at path: {pdf_source}")
        filename = path_obj.name
        stream = open(path_obj, "rb")
    elif isinstance(pdf_source, bytes):
        stream = io.BytesIO(pdf_source)
    else:
        stream = pdf_source
        filename = getattr(pdf_source, "name", "document.pdf")

    try:
        reader = PdfReader(stream)
        total_pages = len(reader.pages)
        pages_to_read = min(total_pages, max_pages)

        page_contents = []
        full_text_parts = []

        for idx in range(pages_to_read):
            page = reader.pages[idx]
            page_text = page.extract_text() or ""
            page_contents.append({
                "page_num": idx + 1,
                "text": page_text.strip(),
            })
            if page_text.strip():
                full_text_parts.append(f"[Página {idx + 1}]\n{page_text.strip()}")

        meta = {}
        if reader.metadata:
            for k, v in reader.metadata.items():
                if v and isinstance(v, str):
                    clean_key = k.lstrip("/")
                    meta[clean_key] = v

        combined_text = "\n\n".join(full_text_parts)

        return {
            "filename": filename,
            "total_pages": total_pages,
            "pages_extracted": pages_to_read,
            "text": combined_text,
            "page_contents": page_contents,
            "metadata": meta,
            "char_count": len(combined_text),
        }
    finally:
        if isinstance(pdf_source, (str, Path)) and not stream.closed:
            stream.close()


def format_pdf_context_for_prompt(pdf_info: dict, max_chars: int = 25000) -> str:
    """Format extracted PDF text as a structured system/human context block."""
    filename = pdf_info.get("filename", "documento.pdf")
    total_pages = pdf_info.get("total_pages", 1)
    extracted_text = pdf_info.get("text", "")

    if len(extracted_text) > max_chars:
        extracted_text = (
            extracted_text[:max_chars]
            + f"\n\n[... Contenido truncado a {max_chars} caracteres para optimizar contexto ...]"
        )

    return (
        f"📄 [DOCUMENTO PDF CARGADO: '{filename}' | Total Páginas: {total_pages}]\n"
        f"==================================================================\n"
        f"{extracted_text}\n"
        f"==================================================================\n"
        f"Instrucción: Utiliza la información del documento PDF adjunto arriba como contexto prioritario "
        f"para responder a la consulta del usuario."
    )


@tool
def read_pdf_file(file_path: str) -> str:
    """
    Read and extract the full text and page breakdown of a local PDF document.
    Use this tool whenever the user refers to a local PDF file, wants to analyze, summarize, or extract facts from a PDF.
    """
    print(f"  📄 [EXECUTION: LangChain @tool 'read_pdf_file'] Reading PDF from path='{file_path}'...")
    try:
        pdf_data = extract_text_from_pdf(file_path, max_pages=30)
        return (
            f"[PDF Extracted Successfully: '{pdf_data['filename']}']\n"
            f"- Total Pages    : {pdf_data['total_pages']}\n"
            f"- Extracted Text :\n\n"
            f"{pdf_data['text'][:3000]}..."
        )
    except Exception as exc:
        return f"[PDF Reader Error] Could not read file '{file_path}': {exc}"
