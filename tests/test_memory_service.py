"""Unit tests for the Hindsight long-term memory service (no live server required)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain_app import memory_service
from langchain_app.memory_service import (
    PROFILE_QUERY,
    bank_id_from_config,
    build_turn_content,
    gather_user_memories,
    format_memories_for_prompt,
    is_hindsight_available,
    is_hindsight_configured,
    recall_memories,
    reflect_on_user_memory,
    remember,
    remember_information,
    resolve_bank_id,
    retain_conversation_turn,
    search_long_term_memory,
)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    """Every test starts with memory configured against a fake URL and a clean cache."""
    monkeypatch.setenv("HINDSIGHT_API_URL", "http://hindsight.test:8888")
    monkeypatch.delenv("HINDSIGHT_ENABLED", raising=False)
    monkeypatch.delenv("HINDSIGHT_BANK_ID", raising=False)
    monkeypatch.delenv("HINDSIGHT_BANK_PREFIX", raising=False)
    monkeypatch.delenv("HINDSIGHT_API_KEY", raising=False)
    memory_service._health_cache.update({"url": None, "ok": False, "checked_at": 0.0})
    memory_service._known_banks.clear()
    yield


def _fake_client(**overrides):
    client = MagicMock()
    client.acreate_bank = AsyncMock(return_value=None)
    client.aupdate_bank_config = AsyncMock(return_value=None)
    client.aretain = AsyncMock(return_value=SimpleNamespace(success=True))
    client.arecall = AsyncMock(return_value=SimpleNamespace(results=[]))
    client.areflect = AsyncMock(return_value=SimpleNamespace(text=""))
    for name, value in overrides.items():
        setattr(client, name, value)
    return client


# ---------------------------------------------------------------------------
# Configuration & availability
# ---------------------------------------------------------------------------
def test_is_hindsight_configured_respects_url_and_kill_switch(monkeypatch):
    assert is_hindsight_configured() is True

    monkeypatch.setenv("HINDSIGHT_ENABLED", "false")
    assert is_hindsight_configured() is False

    monkeypatch.setenv("HINDSIGHT_ENABLED", "true")
    monkeypatch.setenv("HINDSIGHT_API_URL", "")
    assert is_hindsight_configured() is False


@patch("langchain_app.memory_service.requests.get")
def test_is_hindsight_available_caches_health_check(mock_get):
    mock_get.return_value = SimpleNamespace(status_code=200)
    assert is_hindsight_available() is True
    assert is_hindsight_available() is True
    assert mock_get.call_count == 1  # second call served from cache
    assert mock_get.call_args.args[0] == "http://hindsight.test:8888/health"


@patch("langchain_app.memory_service.requests.get")
def test_is_hindsight_available_false_when_server_down(mock_get):
    import requests

    mock_get.side_effect = requests.ConnectionError("refused")
    assert is_hindsight_available(force=True) is False


# ---------------------------------------------------------------------------
# Bank resolution
# ---------------------------------------------------------------------------
def test_resolve_bank_id_prefers_supabase_uuid():
    bank = resolve_bank_id({"db_id": "8F1C-AB12", "email": "ana@example.com"})
    assert bank == "lagchain-user-8f1c-ab12"


def test_resolve_bank_id_hashes_email_without_leaking_pii():
    bank = resolve_bank_id({"email": "Ana@Example.com"})
    assert bank.startswith("lagchain-user-")
    assert "ana" not in bank and "example" not in bank
    # Case-insensitive & stable
    assert bank == resolve_bank_id({"email": "ana@example.com"})


def test_resolve_bank_id_defaults_and_env_override(monkeypatch):
    assert resolve_bank_id(None) == "lagchain-default"
    monkeypatch.setenv("HINDSIGHT_BANK_ID", "Mi Banco CLI")
    assert resolve_bank_id(None) == "mi-banco-cli"


def test_bank_id_from_config():
    assert bank_id_from_config({"configurable": {"memory_bank_id": "user-42"}}) == "user-42"
    assert bank_id_from_config({}) == "lagchain-default"
    assert bank_id_from_config(None) == "lagchain-default"


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def test_format_memories_for_prompt():
    assert format_memories_for_prompt([]) == ""
    block = format_memories_for_prompt([
        {"text": "El usuario es médico en Guayaquil", "type": "world", "occurred": "2026-11-08T10:00:00"},
        {"text": "Prefiere respuestas breves", "type": "observation", "occurred": None},
    ])
    assert "MEMORIA A LARGO PLAZO" in block
    assert "[hecho · 2026-11-08] El usuario es médico en Guayaquil" in block
    assert "[observación] Prefiere respuestas breves" in block


def test_build_turn_content_truncates_long_inputs():
    content = build_turn_content("x" * 5000 + " ¿pregunta final?", "y" * 5000)
    assert content.startswith("Usuario: …")
    assert content.split("\n")[0].endswith("¿pregunta final?")  # tail of user text is kept
    assert "Asistente: " in content
    assert len(content) < 4000


# ---------------------------------------------------------------------------
# Recall / retain / reflect against a mocked client
# ---------------------------------------------------------------------------
def test_recall_memories_returns_empty_when_unavailable():
    with patch("langchain_app.memory_service.is_hindsight_available", return_value=False):
        assert recall_memories("hola") == []


def test_recall_memories_maps_results():
    results = [
        SimpleNamespace(text="El color favorito del usuario es el azul", type="world",
                        occurred_start=None, mentioned_at="2026-11-08", entities=["usuario"]),
        SimpleNamespace(text="   ", type="world", occurred_start=None, mentioned_at=None, entities=None),
    ]
    client = _fake_client(arecall=AsyncMock(return_value=SimpleNamespace(results=results)))
    with patch("langchain_app.memory_service.is_hindsight_available", return_value=True), \
         patch.object(memory_service._runtime, "get_client", return_value=client):
        memories = recall_memories("¿color favorito?", bank_id="bank-1")

    assert memories == [{
        "text": "El color favorito del usuario es el azul",
        "type": "world",
        "occurred": "2026-11-08",
        "entities": ["usuario"],
    }]
    kwargs = client.arecall.call_args.kwargs
    assert kwargs["bank_id"] == "bank-1"
    assert kwargs["budget"] == "low"


def test_recall_memories_unknown_bank_is_empty_not_error():
    not_found = Exception("Not Found")
    not_found.status = 404
    client = _fake_client(arecall=AsyncMock(side_effect=not_found))
    with patch("langchain_app.memory_service.is_hindsight_available", return_value=True), \
         patch.object(memory_service._runtime, "get_client", return_value=client):
        assert recall_memories("hola", bank_id="new-user") == []


def test_retain_conversation_turn_waits_and_configures_bank():
    client = _fake_client()
    with patch("langchain_app.memory_service.is_hindsight_available", return_value=True), \
         patch.object(memory_service._runtime, "get_client", return_value=client):
        ok = retain_conversation_turn(
            "Me llamo Mario", "¡Hola Mario!", bank_id="bank-1", thread_id="th-1", intent="search", wait=True,
        )

    assert ok is True
    client.acreate_bank.assert_awaited_once()
    client.aupdate_bank_config.assert_awaited_once_with(
        "bank-1", retain_strategies={"nota": {"retain_extraction_mode": "chunks"}}
    )
    kwargs = client.aretain.call_args.kwargs
    assert kwargs["bank_id"] == "bank-1"
    assert kwargs["content"] == "Usuario: Me llamo Mario\nAsistente: ¡Hola Mario!"
    assert kwargs["metadata"]["thread_id"] == "th-1"
    assert kwargs["retain_async"] is False


def test_retain_conversation_turn_background_and_flush():
    client = _fake_client()
    with patch("langchain_app.memory_service.is_hindsight_available", return_value=True), \
         patch.object(memory_service._runtime, "get_client", return_value=client):
        assert retain_conversation_turn("Prefiero Python", "Anotado", bank_id="bank-2") is True
        memory_service.flush_pending_memories(timeout=5)

    assert client.aretain.call_args.kwargs["retain_async"] is True


def test_retain_skipped_when_unavailable():
    with patch("langchain_app.memory_service.is_hindsight_available", return_value=False):
        assert retain_conversation_turn("hola", "hola") is False


def test_remember_reports_unconfigured(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_URL", "")
    ok, message = remember("mi color favorito es el azul")
    assert ok is False
    assert "HINDSIGHT_API_URL" in message


# ---------------------------------------------------------------------------
# LangChain tools (bank id injected via RunnableConfig)
# ---------------------------------------------------------------------------
def test_remember_information_tool_uses_bank_from_config():
    with patch("langchain_app.memory_service.remember", return_value=(True, "Guardado")) as mock_remember:
        out = remember_information.invoke(
            {"information": "Recuerda que soy vegetariano"},
            config={"configurable": {"memory_bank_id": "user-7"}},
        )
    assert out == "Guardado"
    mock_remember.assert_called_once_with("Recuerda que soy vegetariano", bank_id="user-7")


def test_memory_tools_hide_config_from_llm_schema():
    for t in (remember_information, search_long_term_memory, reflect_on_user_memory):
        props = t.tool_call_schema.model_json_schema()["properties"]
        assert "config" not in props


def test_reflect_tool_gathers_question_and_profile_memories():
    memories = [{"text": "Eres médico en Guayaquil", "type": "world", "occurred": None, "entities": []}]
    with patch("langchain_app.memory_service.is_hindsight_available", return_value=True), \
         patch("langchain_app.memory_service.recall_memories", return_value=memories) as mock_recall, \
         patch("langchain_app.memory_service.reflect_on_memories") as mock_reflect:
        out = reflect_on_user_memory.invoke(
            {"question": "¿Qué sabes de mí?"},
            config={"configurable": {"memory_bank_id": "user-9"}},
        )
    assert "Eres médico en Guayaquil" in out
    assert out.count("Eres médico en Guayaquil") == 1  # deduplicated across both recalls
    queries = [c.args[0] for c in mock_recall.call_args_list]
    assert queries[0] == "¿Qué sabes de mí?"
    assert queries[1] == PROFILE_QUERY
    assert all(c.kwargs["bank_id"] == "user-9" for c in mock_recall.call_args_list)
    mock_reflect.assert_not_called()  # Hindsight reflect is opt-in (HINDSIGHT_REFLECT_ENABLED)


def test_reflect_tool_includes_hindsight_reflect_when_enabled(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_REFLECT_ENABLED", "true")
    with patch("langchain_app.memory_service.is_hindsight_available", return_value=True), \
         patch("langchain_app.memory_service.recall_memories", return_value=[]), \
         patch("langchain_app.memory_service.reflect_on_memories", return_value="Síntesis de Hindsight"):
        out = reflect_on_user_memory.invoke({"question": "¿Qué sabes de mí?"}, config={"configurable": {}})
    assert "Síntesis de Hindsight" in out


def test_search_long_term_memory_tool_reports_missing_memories():
    with patch("langchain_app.memory_service.is_hindsight_available", return_value=True), \
         patch("langchain_app.memory_service.recall_memories", return_value=[]):
        out = search_long_term_memory.invoke({"query": "¿cómo me llamo?"}, config={"configurable": {}})
    assert out.startswith("SIN MEMORIAS")


def test_gather_user_memories_adds_profile_when_targeted_recall_is_thin():
    targeted = [{"text": "Vive en Quito", "type": "world", "occurred": None, "entities": []}]
    profile = [
        {"text": "Vive en Quito", "type": "world", "occurred": None, "entities": []},
        {"text": "Es cardióloga", "type": "world", "occurred": None, "entities": []},
    ]
    with patch("langchain_app.memory_service.recall_memories", side_effect=[targeted, profile]) as mock_recall:
        memories = gather_user_memories("¿dónde vivo?", bank_id="b")
    assert [m["text"] for m in memories] == ["Vive en Quito", "Es cardióloga"]
    assert mock_recall.call_args_list[1].args[0] == PROFILE_QUERY


def test_gather_user_memories_skips_profile_when_targeted_recall_is_rich():
    rich = [{"text": f"hecho {i}", "type": "world", "occurred": None, "entities": []} for i in range(4)]
    with patch("langchain_app.memory_service.recall_memories", return_value=rich) as mock_recall:
        memories = gather_user_memories("¿qué proyectos tengo?", bank_id="b")
    assert len(memories) == 4
    assert mock_recall.call_count == 1


def test_retain_turn_with_note_stores_user_words_verbatim_first():
    client = _fake_client()
    with patch("langchain_app.memory_service.is_hindsight_available", return_value=True), \
         patch.object(memory_service._runtime, "get_client", return_value=client):
        ok = retain_conversation_turn(
            "Me llamo Mario y vivo en Guayaquil", "¡Encantado!", bank_id="b", intent="user_memory",
            wait=True, save_note=True, include_assistant=False,
        )
    assert ok is True
    note_call, extraction_call = client.aretain.call_args_list
    assert note_call.kwargs["strategy"] == "nota"
    assert "Me llamo Mario y vivo en Guayaquil" in note_call.kwargs["content"]
    assert note_call.kwargs["retain_async"] is False
    assert "strategy" not in extraction_call.kwargs
    assert extraction_call.kwargs["content"] == "Usuario: Me llamo Mario y vivo en Guayaquil"


def test_remember_stores_note_now_and_extracts_in_background():
    client = _fake_client()
    with patch("langchain_app.memory_service.is_hindsight_available", return_value=True), \
         patch.object(memory_service._runtime, "get_client", return_value=client):
        ok, message = remember("Recuerda que soy alérgico a la penicilina", bank_id="b")
        memory_service.flush_pending_memories(timeout=5)
    assert ok is True
    assert "inmediato" in message
    note_call, extraction_call = client.aretain.call_args_list
    assert note_call.kwargs["strategy"] == "nota"
    assert extraction_call.kwargs["retain_async"] is True


def test_remember_falls_back_to_extraction_when_note_fails():
    client = _fake_client(aretain=AsyncMock(side_effect=[RuntimeError("unknown strategy"), SimpleNamespace(success=True)]))
    with patch("langchain_app.memory_service.is_hindsight_available", return_value=True), \
         patch.object(memory_service._runtime, "get_client", return_value=client):
        ok, message = remember("Recuerda que mi color favorito es el verde", bank_id="b")
    assert ok is True
    assert client.aretain.call_args_list[1].kwargs["retain_async"] is True


def test_search_long_term_memory_tool_formats_results():
    memories = [{"text": "Vive en Quito", "type": "world", "occurred": None, "entities": []}]
    with patch("langchain_app.memory_service.is_hindsight_available", return_value=True), \
         patch("langchain_app.memory_service.recall_memories", return_value=memories):
        out = search_long_term_memory.invoke({"query": "¿dónde vive?"}, config={"configurable": {}})
    assert "Vive en Quito" in out
