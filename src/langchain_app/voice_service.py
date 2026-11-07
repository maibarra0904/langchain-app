import base64
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal

import httpx
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
PHONON_MODEL = "phonon:40m-multilingual"
AUDIO_DIR = Path(__file__).resolve().parents[2] / "audio_outputs"


def get_audio_dir() -> Path:
    """Ensure audio output directory exists and return it."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    return AUDIO_DIR


def is_phonon_available() -> bool:
    """Check if Ollama is running and has the phonon:40m-multilingual model."""
    try:
        resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3.0)
        if resp.status_code == 200:
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            return any("phonon:40m-multilingual" in m for m in models)
    except Exception:
        pass
    return False


def clean_text_for_speech(text: str) -> str:
    """Strip markdown formatting, code blocks, and math delimiters for natural TTS pronunciation."""
    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", " [código omitido en audio] ", text)
    # Remove inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Remove LaTeX math blocks and symbols ($$...$$, \(...\), etc.)
    text = re.sub(r"\$\$[\s\S]*?\$\$", "", text)
    text = re.sub(r"\$([^\$]+)\$", r"\1", text)
    text = re.sub(r"\\\[[\s\S]*?\\\]", "", text)
    text = re.sub(r"\\\(|\\\)", "", text)
    # Remove markdown headers and list bullets
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


@traceable(run_type="tool", name="phonon_tts_synthesis", tags=["tts", "phonon", "ollama"])
def synthesize_speech(
    text: str,
    voice: str = "Alba",
    output_filename: str | None = None,
    play_audio: bool = False,
) -> tuple[bytes, Path]:
    """
    Synthesize text into speech using Ollama's phonon:40m-multilingual model.
    Returns (audio_bytes, output_path).
    """
    cleaned = clean_text_for_speech(text)
    if not cleaned:
        cleaned = "Respuesta vacía."

    url = f"{OLLAMA_BASE_URL}/v1/audio/speech"
    payload = {
        "model": PHONON_MODEL,
        "input": cleaned[:1500],  # Limit speech chunk size for optimal performance
        "voice": voice,
    }

    try:
        response = httpx.post(url, json=payload, timeout=45.0)
        response.raise_for_status()
        audio_bytes = response.content
    except Exception as exc:
        raise RuntimeError(f"Error communicating with Ollama phonon TTS: {exc}") from exc

    output_dir = get_audio_dir()
    filename = output_filename or f"speech_{os.urandom(4).hex()}.wav"
    output_path = output_dir / filename

    with open(output_path, "wb") as f:
        f.write(audio_bytes)

    if play_audio and sys.platform == "darwin" and shutil.which("afplay"):
        try:
            subprocess.Popen(["afplay", str(output_path)])
        except Exception:
            pass

    return audio_bytes, output_path


@traceable(run_type="retriever", name="gemini_audio_transcription", tags=["stt", "speech-input"])
def transcribe_audio(audio_data: bytes, mime_type: str = "audio/wav") -> str:
    """Transcribe spoken audio bytes into text using Google Gemini's multimodal capabilities."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "tu_api_key_aqui":
        raise ValueError("GOOGLE_API_KEY is required in .env for audio transcription.")

    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0)

    audio_b64 = base64.b64encode(audio_data).decode("utf-8")
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Transcribe the spoken words in this audio exactly into text in the original spoken language. Return only the transcript without quotes, commentary, or conversational filler.",
            },
            {
                "type": "media",
                "mime_type": mime_type,
                "data": audio_b64,
            },
        ]
    )

    response = llm.invoke([message])
    content = response.content
    if isinstance(content, list):
        parts = [item.get("text", "") if isinstance(item, dict) else str(item) for item in content]
        return " ".join(parts).strip()
    return str(content).strip()


@tool
def speak_response(text: str, voice: str = "Alba") -> str:
    """
    Generate spoken voice audio for a text response using Ollama's phonon:40m-multilingual model.
    Use this tool whenever the user asks for a spoken response, to speak aloud, read the answer, or produce audio speech.
    """
    print(f"  🎙️ [EXECUTION: LangChain @tool 'speak_response'] Synthesizing voice with phonon:40m (voice='{voice}')...")
    try:
        audio_bytes, output_path = synthesize_speech(text, voice=voice, play_audio=True)
        return (
            f"[Voice Generated Successfully with Ollama phonon:40m-multilingual]\n"
            f"- Voice        : {voice}\n"
            f"- Audio File   : {output_path}\n"
            f"- Audio Size   : {len(audio_bytes)} bytes\n"
            f"- Spoken Text  : {text[:200]}..."
        )
    except Exception as exc:
        return f"[Voice Generation Error] Could not synthesize speech with phonon: {exc}"

