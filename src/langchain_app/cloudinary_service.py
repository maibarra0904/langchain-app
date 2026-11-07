import io
import os
import re
import uuid
from pathlib import Path

import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
from langchain_core.tools import tool
from langsmith import traceable

load_dotenv(override=True)


def is_cloudinary_configured() -> bool:
    """Check if Cloudinary credentials are set in environment variables."""
    url = os.getenv("CLOUDINARY_URL", "")
    if url and url.startswith("cloudinary://"):
        return True

    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME", "").strip()
    api_key = os.getenv("CLOUDINARY_API_KEY", "").strip()
    api_secret = os.getenv("CLOUDINARY_API_SECRET", "").strip()

    placeholders = {"tu_cloud_name", "tu_cloudinary_api_key", "tu_cloudinary_api_secret", ""}
    return (
        cloud_name not in placeholders
        and api_key not in placeholders
        and api_secret not in placeholders
    )


def configure_cloudinary() -> bool:
    """Initialize Cloudinary SDK with environment settings."""
    if not is_cloudinary_configured():
        return False

    url = os.getenv("CLOUDINARY_URL", "")
    if url and url.startswith("cloudinary://"):
        # cloudinary automatically parses CLOUDINARY_URL if present
        return True

    cloudinary.config(
        cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
        api_key=os.getenv("CLOUDINARY_API_KEY"),
        api_secret=os.getenv("CLOUDINARY_API_SECRET"),
        secure=True,
    )
    return True


@traceable(run_type="tool", name="cloudinary_pdf_upload", tags=["cloudinary", "pdf-upload", "storage"])
def upload_pdf_to_cloudinary(
    file_data: bytes | io.BytesIO | str | Path,
    filename: str | None = None,
    folder: str = "langchain_app_pdfs",
) -> dict:
    """
    Upload a PDF file (from bytes, BytesIO, or disk path) to Cloudinary.

    Returns a dictionary with:
        - success: bool
        - secure_url: HTTPS download/view URL on Cloudinary
        - public_id: Unique asset identifier in Cloudinary
        - bytes: File size in bytes
        - created_at: Timestamp of upload
        - filename: Original or sanitized file name
    """
    if not is_cloudinary_configured():
        raise ValueError(
            "Cloudinary credentials are not configured. Please set CLOUDINARY_CLOUD_NAME, "
            "CLOUDINARY_API_KEY, and CLOUDINARY_API_SECRET in your .env file."
        )

    configure_cloudinary()

    # Determine filename and payload
    if isinstance(file_data, (str, Path)):
        path_obj = Path(file_data)
        if not path_obj.exists():
            raise FileNotFoundError(f"PDF file not found at: '{file_data}'")
        resolved_filename = filename or path_obj.name
        upload_source = str(path_obj)
    elif isinstance(file_data, bytes):
        resolved_filename = filename or f"document_{uuid.uuid4().hex[:6]}.pdf"
        upload_source = io.BytesIO(file_data)
    elif isinstance(file_data, io.BytesIO):
        resolved_filename = filename or f"document_{uuid.uuid4().hex[:6]}.pdf"
        upload_source = file_data
    else:
        raise TypeError(f"Unsupported file_data type: {type(file_data)}")

    # Clean filename for public_id
    base_name = re.sub(r"[^\w\-.]", "_", Path(resolved_filename).stem)
    custom_public_id = f"{base_name}_{uuid.uuid4().hex[:6]}"

    print(f"  ☁️  [EXECUTION: Cloudinary API] Uploading '{resolved_filename}' to folder '{folder}'...")

    try:
        response = cloudinary.uploader.upload(
            upload_source,
            resource_type="raw",
            folder=folder,
            public_id=custom_public_id,
            overwrite=True,
        )
    except Exception as exc:
        raise RuntimeError(f"Error uploading PDF to Cloudinary: {exc}") from exc

    secure_url = response.get("secure_url") or response.get("url", "")
    public_id = response.get("public_id", custom_public_id)
    file_size = response.get("bytes", 0)
    created_at = response.get("created_at", "")

    print(f"  ✅ [EXECUTION: Cloudinary API] Upload success! Public ID: {public_id} -> {secure_url}")

    return {
        "success": True,
        "secure_url": secure_url,
        "public_id": public_id,
        "bytes": file_size,
        "created_at": created_at,
        "filename": resolved_filename,
    }


@tool
def upload_pdf_to_cloud(file_path: str) -> str:
    """
    Upload a local PDF file to Cloudinary cloud storage and return its public HTTPS URL.
    Use this tool whenever the user asks to upload, back up, or host a PDF file on Cloudinary or in the cloud.
    """
    print(f"  🛠️  [EXECUTION: LangChain @tool 'upload_pdf_to_cloud'] Uploading file='{file_path}'...")
    try:
        res = upload_pdf_to_cloudinary(file_path)
        return (
            f"[PDF Subido Exitosamente a Cloudinary]\n"
            f"- Archivo        : {res['filename']}\n"
            f"- Enlace Seguro  : {res['secure_url']}\n"
            f"- Asset ID (MCP) : {res['public_id']}\n"
            f"- Tamaño         : {res['bytes']} bytes"
        )
    except Exception as exc:
        return f"[Cloudinary Upload Error] Could not upload '{file_path}': {exc}"

