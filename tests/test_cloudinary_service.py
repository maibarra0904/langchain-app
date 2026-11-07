from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_app.cloudinary_service import (
    is_cloudinary_configured,
    upload_pdf_to_cloudinary,
    upload_pdf_to_cloud,
)


def test_is_cloudinary_configured_false(monkeypatch):
    monkeypatch.setenv("CLOUDINARY_CLOUD_NAME", "")
    monkeypatch.setenv("CLOUDINARY_API_KEY", "")
    monkeypatch.setenv("CLOUDINARY_API_SECRET", "")
    monkeypatch.delenv("CLOUDINARY_URL", raising=False)
    assert is_cloudinary_configured() is False


def test_is_cloudinary_configured_true(monkeypatch):
    monkeypatch.setenv("CLOUDINARY_CLOUD_NAME", "my_cloud")
    monkeypatch.setenv("CLOUDINARY_API_KEY", "123456789")
    monkeypatch.setenv("CLOUDINARY_API_SECRET", "abcdefsecret")
    assert is_cloudinary_configured() is True


def test_upload_pdf_unconfigured_raises_value_error(monkeypatch):
    monkeypatch.setenv("CLOUDINARY_CLOUD_NAME", "")
    monkeypatch.delenv("CLOUDINARY_URL", raising=False)
    with pytest.raises(ValueError, match="Cloudinary credentials are not configured"):
        upload_pdf_to_cloudinary(b"%PDF-1.4 sample content", filename="test.pdf")


def test_upload_pdf_file_not_found(monkeypatch):
    monkeypatch.setenv("CLOUDINARY_CLOUD_NAME", "test_cloud")
    monkeypatch.setenv("CLOUDINARY_API_KEY", "test_key")
    monkeypatch.setenv("CLOUDINARY_API_SECRET", "test_secret")
    with pytest.raises(FileNotFoundError):
        upload_pdf_to_cloudinary("/path/to/non_existent_file.pdf")


def test_upload_pdf_bytes_mocked(monkeypatch):
    monkeypatch.setenv("CLOUDINARY_CLOUD_NAME", "test_cloud")
    monkeypatch.setenv("CLOUDINARY_API_KEY", "test_key")
    monkeypatch.setenv("CLOUDINARY_API_SECRET", "test_secret")

    fake_response = {
        "secure_url": "https://res.cloudinary.com/test_cloud/raw/upload/v12345/doc.pdf",
        "public_id": "langchain_app_pdfs/doc_1234",
        "bytes": 5042,
        "created_at": "2026-11-07T12:00:00Z",
    }

    with patch("cloudinary.uploader.upload", return_value=fake_response) as mock_upload:
        res = upload_pdf_to_cloudinary(b"%PDF-1.4 mock bytes", filename="doc.pdf")
        assert res["success"] is True
        assert res["secure_url"] == "https://res.cloudinary.com/test_cloud/raw/upload/v12345/doc.pdf"
        assert res["public_id"] == "langchain_app_pdfs/doc_1234"
        assert res["bytes"] == 5042
        assert mock_upload.called


def test_upload_pdf_file_path_mocked(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CLOUDINARY_CLOUD_NAME", "test_cloud")
    monkeypatch.setenv("CLOUDINARY_API_KEY", "test_key")
    monkeypatch.setenv("CLOUDINARY_API_SECRET", "test_secret")

    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 local sample file")

    fake_response = {
        "secure_url": "https://res.cloudinary.com/test_cloud/raw/upload/v12345/sample.pdf",
        "public_id": "langchain_app_pdfs/sample_abcd",
        "bytes": 1024,
        "created_at": "2026-11-07T12:00:00Z",
    }

    with patch("cloudinary.uploader.upload", return_value=fake_response):
        res = upload_pdf_to_cloudinary(str(pdf_file))
        assert res["success"] is True
        assert res["filename"] == "sample.pdf"
        assert res["secure_url"].startswith("https://res.cloudinary.com")


def test_upload_pdf_to_cloud_tool_success(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CLOUDINARY_CLOUD_NAME", "test_cloud")
    monkeypatch.setenv("CLOUDINARY_API_KEY", "test_key")
    monkeypatch.setenv("CLOUDINARY_API_SECRET", "test_secret")

    pdf_file = tmp_path / "factura.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 test invoice")

    fake_response = {
        "secure_url": "https://res.cloudinary.com/test/raw/upload/factura.pdf",
        "public_id": "langchain_app_pdfs/factura_123",
        "bytes": 2048,
        "created_at": "2026-11-07T12:00:00Z",
    }

    with patch("cloudinary.uploader.upload", return_value=fake_response):
        output = upload_pdf_to_cloud.invoke({"file_path": str(pdf_file)})
        assert "[PDF Subido Exitosamente a Cloudinary]" in output
        assert "factura.pdf" in output
        assert "https://res.cloudinary.com/test/raw/upload/factura.pdf" in output


def test_upload_pdf_to_cloud_tool_error(monkeypatch):
    monkeypatch.setenv("CLOUDINARY_CLOUD_NAME", "")
    monkeypatch.delenv("CLOUDINARY_URL", raising=False)
    output = upload_pdf_to_cloud.invoke({"file_path": "/fake/path.pdf"})
    assert "[Cloudinary Upload Error]" in output
