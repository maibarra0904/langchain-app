import pytest
from langchain_app.voice_service import (
    clean_text_for_speech,
    is_phonon_available,
    speak_response,
    synthesize_speech,
)


def test_clean_text_for_speech_removes_markdown():
    raw = "Here is **bold**, *italic*, and `code`. Also visit https://example.com"
    cleaned = clean_text_for_speech(raw)
    assert "https://" not in cleaned
    assert "**" not in cleaned
    assert "`" not in cleaned
    assert "bold" in cleaned


def test_clean_text_for_speech_removes_code_blocks():
    raw = "Explanation:\n```python\nprint('hello')\n```\nDone."
    cleaned = clean_text_for_speech(raw)
    assert "print('hello')" not in cleaned
    assert "Done." in cleaned


def test_is_phonon_available():
    # Model phonon:40m-multilingual is installed in Ollama on this machine
    available = is_phonon_available()
    assert isinstance(available, bool)


def test_synthesize_speech_generation():
    if not is_phonon_available():
        pytest.skip("Ollama with phonon:40m-multilingual is not reachable")

    audio_bytes, path = synthesize_speech("Prueba corta de síntesis", voice="Alba")
    assert len(audio_bytes) > 1000
    assert path.exists()
    assert path.suffix == ".wav"


def test_speak_response_tool():
    assert speak_response.name == "speak_response"
    assert "phonon:40m-multilingual" in speak_response.description

