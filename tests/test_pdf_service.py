import io
from pathlib import Path
import pytest
from pypdf import PdfWriter
from langchain_app.pdf_service import (
    extract_text_from_pdf,
    format_pdf_context_for_prompt,
    read_pdf_file,
)


def _create_sample_pdf_bytes(text_lines: list[str]) -> bytes:
    """Helper to create a valid minimal PDF in memory."""
    writer = PdfWriter()
    # Add a blank page
    writer.add_blank_page(width=300, height=300)
    # Add another page
    writer.add_blank_page(width=300, height=300)
    stream = io.BytesIO()
    writer.write(stream)
    return stream.getvalue()


def test_extract_text_from_pdf_bytes():
    pdf_bytes = _create_sample_pdf_bytes(["Test line"])
    result = extract_text_from_pdf(pdf_bytes)
    assert result["total_pages"] == 2
    assert result["pages_extracted"] == 2
    assert isinstance(result["text"], str)
    assert "filename" in result


def test_extract_text_from_pdf_file(tmp_path: Path):
    pdf_file = tmp_path / "sample_report.pdf"
    pdf_bytes = _create_sample_pdf_bytes(["Sample content"])
    pdf_file.write_bytes(pdf_bytes)

    result = extract_text_from_pdf(pdf_file)
    assert result["filename"] == "sample_report.pdf"
    assert result["total_pages"] == 2


def test_extract_text_file_not_found():
    with pytest.raises(FileNotFoundError):
        extract_text_from_pdf("/non/existent/path/doc.pdf")


def test_format_pdf_context_for_prompt():
    info = {
        "filename": "financial_report.pdf",
        "total_pages": 5,
        "text": "Total revenue for 2026 was $10M.",
    }
    context = format_pdf_context_for_prompt(info)
    assert "DOCUMENTO PDF CARGADO: 'financial_report.pdf'" in context
    assert "Total revenue for 2026 was $10M." in context
    assert "Instrucción" in context


def test_read_pdf_file_tool(tmp_path: Path):
    assert read_pdf_file.name == "read_pdf_file"
    pdf_file = tmp_path / "manual.pdf"
    pdf_file.write_bytes(_create_sample_pdf_bytes(["Manual"]))
    output = read_pdf_file.invoke({"file_path": str(pdf_file)})
    assert "[PDF Extracted Successfully: 'manual.pdf']" in output

