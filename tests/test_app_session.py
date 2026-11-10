"""Streamlit integration tests: the login must survive page reloads (new browser sessions).

Each ``AppTest`` instance is a fresh Streamlit session, exactly like a browser reload. The
session cookie the browser would send is injected through ``st.context.cookies``.
External services (Supabase, LangSmith, Ollama, Hindsight) are stubbed out.
"""

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from streamlit.runtime.context import ContextProxy
from streamlit.testing.v1 import AppTest

from langchain_app import session_service
from langchain_app.session_service import SESSION_COOKIE_NAME, create_session_token, verify_session_token

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")
USER = {
    "google_id": "1234567890",
    "email": "ana@example.com",
    "name": "Ana Pérez",
    "avatar": "https://lh3.googleusercontent.com/a/photo.jpg",
    "db_id": "9057d70f-8208-4c88-b4aa-e98186bbf752",
    "authenticated": True,
    "auth_method": "google_oauth",
}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("SESSION_SECRET", "test-secret-with-enough-entropy-0123456789")
    session_service._revoked_jti.clear()
    yield


def _new_session(cookies: dict | None = None) -> tuple[AppTest, ExitStack]:
    """A brand-new browser session (page load/reload) carrying ``cookies``."""
    stack = ExitStack()
    # app.py calls load_dotenv(override=True): keep the test's SESSION_SECRET instead of the real .env one.
    stack.enter_context(patch("dotenv.load_dotenv", return_value=True))
    stack.enter_context(patch.object(ContextProxy, "cookies", new_callable=PropertyMock, return_value=cookies or {}))
    stack.enter_context(patch("langchain_app.supabase_service.is_supabase_configured", return_value=False))
    stack.enter_context(patch("langchain_app.cloudinary_service.is_cloudinary_configured", return_value=False))
    stack.enter_context(patch("langchain_app.main.get_langsmith_client", return_value=None))
    stack.enter_context(patch("langchain_app.main.build_chat_graph", return_value=MagicMock()))
    stack.enter_context(patch("langchain_app.voice_service.is_phonon_available", return_value=False))
    stack.enter_context(patch(
        "langchain_app.memory_service.get_memory_status",
        return_value={"configured": False, "available": False, "url": None, "ui_url": None},
    ))
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    return at, stack


def _on_login_screen(at: AppTest) -> bool:
    return any("Iniciar Sesión" in s.value for s in at.subheader)


def _scripts(at: AppTest) -> str:
    return "\n".join(h.proto.body for h in at.get("html"))


def test_reload_without_cookie_shows_login():
    at, stack = _new_session()
    with stack:
        at.run()
    assert not at.exception
    assert _on_login_screen(at)
    assert at.session_state["user"] is None


def test_reload_with_valid_cookie_restores_the_session():
    token = create_session_token(USER)
    at, stack = _new_session({SESSION_COOKIE_NAME: token})
    with stack:
        at.run()
    assert not at.exception
    assert not _on_login_screen(at)
    user = at.session_state["user"]
    assert user["email"] == USER["email"] and user["db_id"] == USER["db_id"]


def test_tampered_or_expired_cookie_shows_login():
    for bad in ["garbage", create_session_token(USER, ttl_seconds=-1)]:
        at, stack = _new_session({SESSION_COOKIE_NAME: bad})
        with stack:
            at.run()
        assert _on_login_screen(at), bad


def test_login_sets_the_persistent_cookie():
    at, stack = _new_session()
    with stack:
        at.run()
        next(b for b in at.button if "Entrar al Agente" in b.label).click().run()

    assert not at.exception
    assert not _on_login_screen(at)
    scripts = _scripts(at)
    assert f"document.cookie = '{SESSION_COOKIE_NAME}=" in scripts
    token = scripts.split(f"{SESSION_COOKIE_NAME}=")[1].split(";")[0]
    assert verify_session_token(token)["email"] == "usuario@gmail.com"


def test_logout_clears_cookie_and_revokes_token():
    token = create_session_token(USER)
    at, stack = _new_session({SESSION_COOKIE_NAME: token})
    with stack:
        at.run()
        next(b for b in at.sidebar.button if "Cerrar sesión" in b.label).click().run()
        # Same browser session: st.context.cookies still holds the stale token, it must not log back in.
        at.run()

    assert _on_login_screen(at)
    assert at.session_state["user"] is None
    assert "max-age=0" in _scripts(at)
    assert verify_session_token(token) is None  # revoked server-side

    # A later reload (e.g. if the browser kept the cookie) must not restore the session either.
    at2, stack2 = _new_session({SESSION_COOKIE_NAME: token})
    with stack2:
        at2.run()
    assert _on_login_screen(at2)
