"""Hindsight Long-Term Memory Service for the LangGraph agent.

Integra Hindsight (https://github.com/vectorize-io/hindsight) como memoria a largo
plazo del agente. Las tres operaciones de Hindsight se mapean así:

- **retain**: tras cada turno relevante se guarda el intercambio en el banco de
  memoria del usuario. Hindsight extrae hechos, entidades y datos temporales, y
  consolida observaciones en segundo plano.
- **recall**: antes de responder se recuperan las memorias relevantes para la
  consulta (búsqueda semántica + BM25 + grafo + temporal) y se inyectan como contexto.
- **reflect**: responde preguntas sobre lo que el agente sabe del usuario
  razonando sobre todas sus memorias ("¿qué sabes de mí?").

Hay un banco por usuario (identidad de Google/Supabase), por lo que las memorias
nunca se mezclan entre usuarios. El servicio degrada con elegancia: si
``HINDSIGHT_API_URL`` no está configurada o el servidor no responde, el agente
sigue funcionando exactamente igual que antes.

Nota técnica: ``hindsight-client`` usa aiohttp y su sesión queda ligada al event
loop donde se creó. Como el grafo se invoca en modo síncrono desde hilos distintos
(Streamlit, CLI), todas las llamadas se ejecutan en un único event loop dedicado
que vive en un hilo daemon propio.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import hashlib
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Coroutine, Optional

import requests
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langsmith import traceable

load_dotenv()

logger = logging.getLogger("langchain_app.memory_service")

PLACEHOLDER_URLS = {"", "http://tu_hindsight_url"}
DEFAULT_BANK_PREFIX = "lagchain"
HEALTH_CACHE_SECONDS = 30.0
RECALL_TIMEOUT_SECONDS = 20.0
RETAIN_TIMEOUT_SECONDS = 90.0
REFLECT_TIMEOUT_SECONDS = 120.0
NOTE_TIMEOUT_SECONDS = 30.0
NOTE_STRATEGY = "nota"
PROFILE_QUERY = (
    "Perfil del usuario: nombre, profesión, ubicación, preferencias, gustos, "
    "proyectos, objetivos, salud y documentos compartidos"
)
MAX_USER_CHARS = 2000
MAX_ASSISTANT_CHARS = 1500

# Instrucciones del banco: guían qué extrae Hindsight y cómo razona en reflect.
BANK_RETAIN_MISSION = (
    "Extrae hechos duraderos y útiles sobre el usuario: identidad, profesión, ubicación, "
    "preferencias, objetivos, proyectos, decisiones, documentos que ha compartido y datos "
    "relevantes de esos documentos (por ejemplo, valores de exámenes médicos). "
    "Ignora saludos, cortesías y resultados genéricos de búsquedas web que no digan nada del usuario."
)
BANK_REFLECT_MISSION = (
    "Eres la memoria a largo plazo de un asistente personal. Responde en español, de forma clara, "
    "basándote únicamente en las memorias del usuario. Si no hay evidencia suficiente, dilo explícitamente."
)


# -----------------------------------------------------------------------------
# Configuración
# -----------------------------------------------------------------------------
def get_hindsight_url() -> str:
    """Return the configured Hindsight API base URL (empty string when unset)."""
    return os.getenv("HINDSIGHT_API_URL", "").strip().rstrip("/")


def is_hindsight_configured() -> bool:
    """Return True when a Hindsight server URL is configured and memory is not disabled."""
    if os.getenv("HINDSIGHT_ENABLED", "true").strip().lower() in {"0", "false", "no", "off"}:
        return False
    return get_hindsight_url() not in PLACEHOLDER_URLS


def _get_api_key() -> Optional[str]:
    key = os.getenv("HINDSIGHT_API_KEY", "").strip()
    return key or None


def get_recall_budget() -> str:
    budget = os.getenv("HINDSIGHT_RECALL_BUDGET", "low").strip().lower()
    return budget if budget in {"low", "mid", "high"} else "low"


def get_recall_max_tokens() -> int:
    try:
        return max(256, int(os.getenv("HINDSIGHT_RECALL_MAX_TOKENS", "1024")))
    except ValueError:
        return 1024


# -----------------------------------------------------------------------------
# Disponibilidad (health check cacheado)
# -----------------------------------------------------------------------------
_health_cache: dict[str, Any] = {"url": None, "ok": False, "checked_at": 0.0}
_health_lock = threading.Lock()


def is_hindsight_available(force: bool = False) -> bool:
    """Return True if the Hindsight server answers its health check (cached for 30s)."""
    if not is_hindsight_configured():
        return False

    url = get_hindsight_url()
    now = time.monotonic()
    with _health_lock:
        if (
            not force
            and _health_cache["url"] == url
            and now - _health_cache["checked_at"] < HEALTH_CACHE_SECONDS
        ):
            return bool(_health_cache["ok"])

    ok = False
    try:
        headers = {"Authorization": f"Bearer {_get_api_key()}"} if _get_api_key() else {}
        resp = requests.get(f"{url}/health", timeout=2, headers=headers)
        ok = resp.status_code == 200
    except requests.RequestException as exc:
        logger.debug("Hindsight health check failed: %s", exc)

    with _health_lock:
        _health_cache.update({"url": url, "ok": ok, "checked_at": time.monotonic()})
    return ok


def get_memory_status(force: bool = False) -> dict[str, Any]:
    """Summarize the memory subsystem status for UIs and CLI diagnostics."""
    configured = is_hindsight_configured()
    return {
        "configured": configured,
        "available": is_hindsight_available(force=force) if configured else False,
        "url": get_hindsight_url() or None,
        "ui_url": os.getenv("HINDSIGHT_UI_URL", "").strip() or None,
    }


# -----------------------------------------------------------------------------
# Event loop dedicado + cliente Hindsight
# -----------------------------------------------------------------------------
class _HindsightRuntime:
    """Owns a background asyncio loop and the Hindsight client bound to it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._client: Any = None
        self._client_key: Optional[tuple[str, Optional[str]]] = None
        self._pending: set[concurrent.futures.Future] = set()

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None or self._thread is None or not self._thread.is_alive():
                self._loop = asyncio.new_event_loop()
                self._thread = threading.Thread(
                    target=self._loop.run_forever,
                    name="hindsight-memory-loop",
                    daemon=True,
                )
                self._thread.start()
                # A fresh loop needs a fresh client (aiohttp sessions are loop-bound).
                self._client = None
                self._client_key = None
            return self._loop

    def get_client(self) -> Any:
        """Return a Hindsight client for the current URL/API key (created lazily)."""
        from hindsight_client import Hindsight

        self._ensure_loop()
        key = (get_hindsight_url(), _get_api_key())
        with self._lock:
            if self._client is None or self._client_key != key:
                kwargs: dict[str, Any] = {
                    "base_url": key[0],
                    "timeout": REFLECT_TIMEOUT_SECONDS,
                    "user_agent": "langchain-app-memory/0.1",
                }
                if key[1]:
                    kwargs["api_key"] = key[1]
                self._client = Hindsight(**kwargs)
                self._client_key = key
            return self._client

    def run(self, coro: Coroutine[Any, Any, Any], timeout: float) -> Any:
        """Run a coroutine on the memory loop and block until it finishes."""
        future = asyncio.run_coroutine_threadsafe(coro, self._ensure_loop())
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise

    def submit(self, coro: Coroutine[Any, Any, Any]) -> concurrent.futures.Future:
        """Schedule a coroutine on the memory loop without waiting (fire-and-forget)."""
        future = asyncio.run_coroutine_threadsafe(coro, self._ensure_loop())
        with self._lock:
            self._pending.add(future)

        def _done(f: concurrent.futures.Future) -> None:
            with self._lock:
                self._pending.discard(f)
            if not f.cancelled() and f.exception() is not None:
                logger.warning("Hindsight background retain failed: %s", f.exception())

        future.add_done_callback(_done)
        return future

    def flush(self, timeout: float = 10.0) -> int:
        """Wait for pending background operations. Returns how many were still pending."""
        with self._lock:
            pending = list(self._pending)
        if pending:
            concurrent.futures.wait(pending, timeout=timeout)
        return len(pending)

    def shutdown(self, timeout: float = 5.0) -> None:
        """Flush pending retains and close the aiohttp session cleanly (called at exit)."""
        self.flush(timeout=timeout)
        with self._lock:
            client, loop = self._client, self._loop
            self._client = None
        if client is not None and loop is not None and loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(client.aclose(), loop).result(timeout=2.0)
            except Exception as exc:
                logger.debug("Error closing Hindsight client: %s", exc)


_runtime = _HindsightRuntime()
_known_banks: set[str] = set()
atexit.register(_runtime.shutdown)


def flush_pending_memories(timeout: float = 10.0) -> int:
    """Block until background retains finish (useful before a one-shot CLI process exits)."""
    return _runtime.flush(timeout=timeout)


# -----------------------------------------------------------------------------
# Bancos de memoria
# -----------------------------------------------------------------------------
def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-")[:64]


def resolve_bank_id(user: Optional[dict[str, Any]] = None) -> str:
    """Map an authenticated user to an isolated Hindsight bank id.

    Uses the Supabase UUID when available; otherwise a SHA-256 digest of the
    Google subject/email so that no raw PII ends up in URLs or logs.
    """
    prefix = _slug(os.getenv("HINDSIGHT_BANK_PREFIX", DEFAULT_BANK_PREFIX)) or DEFAULT_BANK_PREFIX
    if user:
        if user.get("db_id"):
            return f"{prefix}-user-{_slug(str(user['db_id']))}"
        identity = user.get("google_id") or user.get("email")
        if identity:
            digest = hashlib.sha256(str(identity).strip().lower().encode("utf-8")).hexdigest()[:16]
            return f"{prefix}-user-{digest}"
    return _slug(os.getenv("HINDSIGHT_BANK_ID", "")) or f"{prefix}-default"


def bank_id_from_config(config: Optional[RunnableConfig]) -> str:
    """Read the memory bank id injected in the LangGraph RunnableConfig."""
    configurable = (config or {}).get("configurable", {}) or {}
    return configurable.get("memory_bank_id") or resolve_bank_id(None)


async def _ensure_bank(client: Any, bank_id: str) -> None:
    """Create/update the bank once per process with this app's missions and the ``nota`` strategy.

    The ``nota`` retain strategy uses Hindsight's ``chunks`` extraction mode: the text is stored
    verbatim (embeddings only, no LLM), so it is recallable within ~1s even if the extraction
    LLM is slow, rate-limited or down.
    """
    if bank_id in _known_banks:
        return
    try:
        await client.acreate_bank(
            bank_id=bank_id,
            retain_mission=BANK_RETAIN_MISSION,
            reflect_mission=BANK_REFLECT_MISSION,
        )
        await client.aupdate_bank_config(
            bank_id, retain_strategies={NOTE_STRATEGY: {"retain_extraction_mode": "chunks"}}
        )
        _known_banks.add(bank_id)
    except Exception as exc:  # Bank config is an optimisation, never block memory on it.
        logger.warning("Could not configure Hindsight bank '%s': %s", bank_id, exc)


# -----------------------------------------------------------------------------
# Recall
# -----------------------------------------------------------------------------
def _memory_to_dict(result: Any) -> dict[str, Any]:
    return {
        "text": getattr(result, "text", "") or "",
        "type": getattr(result, "type", None) or "memory",
        "occurred": getattr(result, "occurred_start", None) or getattr(result, "mentioned_at", None),
        "entities": list(getattr(result, "entities", None) or []),
    }


@traceable(run_type="retriever", name="hindsight_recall", tags=["hindsight", "memory", "recall"])
def recall_memories(
    query: str,
    bank_id: Optional[str] = None,
    max_results: int = 8,
    budget: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Retrieve the long-term memories most relevant to ``query`` (empty list on any failure)."""
    query = (query or "").strip()
    if not query or not is_hindsight_available():
        return []

    bank = bank_id or resolve_bank_id(None)
    # Queries prefixed with a full PDF dump are huge; the question lives at the end.
    if len(query) > 1000:
        query = query[-1000:]

    client = _runtime.get_client()
    try:
        response = _runtime.run(
            client.arecall(
                bank_id=bank,
                query=query,
                budget=budget or get_recall_budget(),
                max_tokens=max_tokens or get_recall_max_tokens(),
            ),
            timeout=RECALL_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        if getattr(exc, "status", None) == 404:
            # New user: the bank is created on the first retain, nothing to recall yet.
            return []
        logger.warning("Hindsight recall failed for bank '%s': %s", bank, exc)
        return []

    memories = [_memory_to_dict(r) for r in (getattr(response, "results", None) or [])]
    return [m for m in memories if m["text"].strip()][:max_results]


def _dedupe(memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique = []
    for m in memories:
        key = re.sub(r"\s+", " ", m.get("text", "")).strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(m)
    return unique


def gather_user_memories(
    question: str,
    bank_id: Optional[str] = None,
    include_profile: bool = False,
    max_results: int = 12,
) -> list[dict[str, Any]]:
    """Collect the evidence needed to answer a question about the user.

    Runs a targeted recall for the question and, when it is thin (or a broad
    "what do you know about me" is asked), a second recall for the user's
    profile, so vague questions still get grounded on who the user is.
    """
    targeted = recall_memories(question, bank_id=bank_id, max_results=max_results)
    memories = list(targeted)
    if include_profile or len(targeted) < 3:
        memories += recall_memories(
            PROFILE_QUERY, bank_id=bank_id, max_results=max_results, budget="mid", max_tokens=2048
        )
    return _dedupe(memories)[:max_results]


TYPE_LABELS = {"world": "hecho", "experience": "experiencia", "observation": "observación"}


def format_memories_for_prompt(memories: list[dict[str, Any]]) -> str:
    """Render recalled memories as a compact prompt block (empty string if none)."""
    if not memories:
        return ""
    lines = []
    for m in memories:
        label = TYPE_LABELS.get(m.get("type") or "", m.get("type") or "memoria")
        when = f" · {str(m['occurred'])[:10]}" if m.get("occurred") else ""
        lines.append(f"- [{label}{when}] {m['text']}")
    return (
        "MEMORIA A LARGO PLAZO DEL USUARIO (Hindsight):\n"
        + "\n".join(lines)
        + "\nEstas memorias son la fuente de verdad sobre el usuario; no inventes datos que no aparezcan aquí."
    )


# -----------------------------------------------------------------------------
# Retain
# -----------------------------------------------------------------------------
def _truncate(text: str, limit: int, keep_tail: bool = False) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return "…" + text[-limit:] if keep_tail else text[:limit] + "…"


def build_turn_content(user_text: str, assistant_text: str) -> str:
    """Format one conversation turn the way Hindsight extracts it best (speaker-labelled)."""
    user_part = _truncate(user_text, MAX_USER_CHARS, keep_tail=True)
    parts = [f"Usuario: {user_part}"]
    if assistant_text and assistant_text.strip():
        parts.append(f"Asistente: {_truncate(assistant_text, MAX_ASSISTANT_CHARS)}")
    return "\n".join(parts)


async def _aretain(
    bank_id: str,
    content: str,
    context: str,
    metadata: dict[str, str],
    retain_async: bool,
    strategy: Optional[str] = None,
    kind: str = "turno",
) -> Any:
    client = _runtime.get_client()
    await _ensure_bank(client, bank_id)
    kwargs: dict[str, Any] = {
        "bank_id": bank_id,
        "content": content,
        "context": context,
        "timestamp": datetime.now(timezone.utc),
        "metadata": metadata,
        "tags": ["source:langchain-app", f"kind:{kind}"],
        "retain_async": retain_async,
    }
    if strategy:
        kwargs["strategy"] = strategy
    return await client.aretain(**kwargs)


async def _aretain_note(bank_id: str, user_text: str, metadata: dict[str, str]) -> Any:
    """Store the user's own words verbatim (no LLM): recallable immediately."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return await _aretain(
        bank_id,
        f"Nota del usuario ({today}): {_truncate(user_text, MAX_USER_CHARS, keep_tail=True)}",
        context="Nota literal del usuario guardada por el asistente",
        metadata=metadata,
        retain_async=False,
        strategy=NOTE_STRATEGY,
        kind="nota",
    )


async def _aretain_turn_job(
    bank_id: str,
    user_text: str,
    assistant_text: str,
    intent: Optional[str],
    metadata: dict[str, str],
    save_note: bool,
    extraction_async: bool,
) -> None:
    if save_note:
        try:
            await _aretain_note(bank_id, user_text, metadata)
        except Exception as exc:  # The LLM extraction below still runs.
            logger.warning("Hindsight note retain failed for bank '%s': %s", bank_id, exc)
    await _aretain(
        bank_id,
        build_turn_content(user_text, assistant_text),
        context=f"Conversación con el asistente LangGraph (intención: {intent or 'general'})",
        metadata=metadata,
        retain_async=extraction_async,
    )


@traceable(run_type="tool", name="hindsight_retain_turn", tags=["hindsight", "memory", "retain"])
def retain_conversation_turn(
    user_text: str,
    assistant_text: str,
    bank_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    intent: Optional[str] = None,
    wait: bool = False,
    save_note: bool = False,
    include_assistant: bool = True,
) -> bool:
    """Store a conversation turn in long-term memory.

    - ``save_note``: also store the user's words verbatim (instantly recallable, no LLM).
    - ``include_assistant``: send the assistant answer to fact extraction too (useful for
      document analyses; noisy for personal statements).

    By default everything runs in the background (and the LLM extraction is processed
    asynchronously server-side), so it never adds latency to the user's answer.
    Returns True if the retain was scheduled/stored.
    """
    if not (user_text or "").strip() or not is_hindsight_available():
        return False

    bank = bank_id or resolve_bank_id(None)
    metadata = {"thread_id": thread_id or "", "intent": intent or "", "app": "langchain-app"}
    coro = _aretain_turn_job(
        bank,
        user_text,
        assistant_text if include_assistant else "",
        intent,
        metadata,
        save_note=save_note,
        extraction_async=not wait,
    )
    if wait:
        try:
            _runtime.run(coro, timeout=RETAIN_TIMEOUT_SECONDS)
            return True
        except Exception as exc:
            logger.warning("Hindsight retain failed for bank '%s': %s", bank, exc)
            return False
    _runtime.submit(coro)
    return True


@traceable(run_type="tool", name="hindsight_remember", tags=["hindsight", "memory", "retain"])
def remember(information: str, bank_id: Optional[str] = None) -> tuple[bool, str]:
    """Store an explicit fact so it can be recalled on the very next turn.

    1. A verbatim note (``nota`` strategy, no LLM) is stored synchronously: available at once.
    2. Hindsight's LLM fact extraction runs in the background to build facts, entities and
       observations from it.
    """
    information = (information or "").strip()
    if not information:
        return False, "No hay información para guardar."
    if not is_hindsight_available():
        return False, _unavailable_message()

    bank = bank_id or resolve_bank_id(None)
    metadata = {"intent": "memory_store", "app": "langchain-app"}
    extraction = dict(
        content=f"Usuario: {information}",
        context="El usuario pidió explícitamente al asistente que recordara esto",
        metadata=metadata,
    )
    try:
        _runtime.run(_aretain_note(bank, information, metadata), timeout=NOTE_TIMEOUT_SECONDS)
    except Exception as exc:
        logger.warning("Hindsight note retain failed, falling back to queued extraction: %s", exc)
        try:
            _runtime.run(_aretain(bank, retain_async=True, **extraction), timeout=NOTE_TIMEOUT_SECONDS)
        except Exception as exc2:
            logger.warning("Hindsight explicit retain failed: %s", exc2)
            return False, f"No se pudo guardar en la memoria de Hindsight: {exc2}"
        return True, "Información enviada a la memoria a largo plazo; estará disponible en cuanto Hindsight la procese."

    _runtime.submit(_aretain(bank, retain_async=True, **extraction))
    return True, "Información guardada en la memoria a largo plazo (disponible de inmediato)."


# -----------------------------------------------------------------------------
# Reflect
# -----------------------------------------------------------------------------
def is_reflect_enabled() -> bool:
    """Hindsight's agentic reflect needs a capable LLM (several calls per question): opt-in."""
    return os.getenv("HINDSIGHT_REFLECT_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}


@traceable(run_type="chain", name="hindsight_reflect", tags=["hindsight", "memory", "reflect"])
def reflect_on_memories(question: str, bank_id: Optional[str] = None) -> Optional[str]:
    """Ask Hindsight to reason over the user's memories. Returns None if unavailable/failed."""
    question = (question or "").strip()
    if not question or not is_hindsight_available():
        return None

    bank = bank_id or resolve_bank_id(None)
    client = _runtime.get_client()
    try:
        response = _runtime.run(
            client.areflect(bank_id=bank, query=question, budget="low"),
            timeout=REFLECT_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        if getattr(exc, "status", None) != 404:
            logger.warning("Hindsight reflect failed for bank '%s': %s", bank, exc)
        return None
    return (getattr(response, "text", "") or "").strip() or None


def _unavailable_message() -> str:
    if not is_hindsight_configured():
        return (
            "La memoria a largo plazo (Hindsight) no está configurada. "
            "Define HINDSIGHT_API_URL en .env y arranca el servidor con ./scripts/start_hindsight.sh."
        )
    return f"El servidor de memoria Hindsight no responde en {get_hindsight_url()}."


NO_MEMORIES_MESSAGE = (
    "SIN MEMORIAS: todavía no hay información guardada sobre el usuario relacionada con esta consulta."
)


# -----------------------------------------------------------------------------
# LangChain tools (el bank_id llega por RunnableConfig: configurable.memory_bank_id)
# -----------------------------------------------------------------------------
@tool
def remember_information(information: str, config: RunnableConfig) -> str:
    """Save a fact, preference or instruction about the user in long-term memory (Hindsight).

    Use it when the user explicitly asks to remember something ("recuerda que...").

    Args:
        information: The information to remember, in the user's own words.
    """
    print(f"  🧠 [EXECUTION: LangChain @tool 'remember_information'] Guardando en Hindsight: '{information[:80]}'")
    _, message = remember(information, bank_id=bank_id_from_config(config))
    return message


@tool
def search_long_term_memory(query: str, config: RunnableConfig) -> str:
    """Search the user's long-term memory (Hindsight) to answer a question about the user.

    Use it for questions about the user's identity, work, projects, preferences or anything
    they shared in previous conversations.

    Args:
        query: The user's question or what to look for in memory.
    """
    print(f"  🧠 [EXECUTION: LangChain @tool 'search_long_term_memory'] Query='{query}'")
    if not is_hindsight_available():
        return _unavailable_message()
    memories = gather_user_memories(query, bank_id=bank_id_from_config(config))
    return format_memories_for_prompt(memories) or NO_MEMORIES_MESSAGE


@tool
def reflect_on_user_memory(question: str, config: RunnableConfig) -> str:
    """Gather everything remembered about the user to answer broad questions like "¿qué sabes de mí?".

    Args:
        question: The question about the user or past conversations.
    """
    print(f"  🧠 [EXECUTION: LangChain @tool 'reflect_on_user_memory'] Question='{question}'")
    if not is_hindsight_available():
        return _unavailable_message()
    bank = bank_id_from_config(config)
    sections = []
    if is_reflect_enabled():
        synthesis = reflect_on_memories(question, bank_id=bank)
        if synthesis:
            sections.append(f"SÍNTESIS DE HINDSIGHT (reflect):\n{synthesis}")
    memories = gather_user_memories(question, bank_id=bank, include_profile=True, max_results=15)
    if memories:
        sections.append(format_memories_for_prompt(memories))
    return "\n\n".join(sections) or NO_MEMORIES_MESSAGE
