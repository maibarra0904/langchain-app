"""Unit tests for the persistent login session service (signed cookie tokens)."""

import base64
import json
import time

import pytest

from langchain_app import session_service
from langchain_app.session_service import (
    SESSION_COOKIE_NAME,
    build_clear_cookie_script,
    build_set_cookie_script,
    create_session_token,
    restore_user_from_cookies,
    revoke_session_token,
    verify_session_token,
)

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
def _secret(monkeypatch):
    monkeypatch.setenv("SESSION_SECRET", "test-secret-with-enough-entropy-0123456789")
    monkeypatch.delenv("SESSION_TTL_DAYS", raising=False)
    session_service._revoked_jti.clear()
    yield


def test_roundtrip_restores_the_same_user():
    token = create_session_token(USER)
    restored = verify_session_token(token)
    assert restored is not None
    for key in ("google_id", "email", "name", "avatar", "db_id", "auth_method"):
        assert restored[key] == USER[key]
    assert restored["authenticated"] is True
    assert restored["restored_from_cookie"] is True


def test_token_is_cookie_and_js_safe():
    token = create_session_token(USER)
    assert all(c.isalnum() or c in "-_." for c in token)
    assert len(token) < 3500  # browsers cap cookies at ~4 KB


def test_tampered_payload_is_rejected():
    token = create_session_token(USER)
    payload_b64, signature = token.split(".")
    payload = json.loads(base64.urlsafe_b64decode(payload_b64 + "=" * (-len(payload_b64) % 4)))
    payload["email"] = "attacker@example.com"
    forged = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    assert verify_session_token(f"{forged}.{signature}") is None


def test_token_signed_with_other_secret_is_rejected(monkeypatch):
    token = create_session_token(USER)
    monkeypatch.setenv("SESSION_SECRET", "a-completely-different-secret-value-9876543210")
    assert verify_session_token(token) is None


def test_expired_token_is_rejected(monkeypatch):
    token = create_session_token(USER, ttl_seconds=60)
    real_time = time.time
    monkeypatch.setattr(session_service.time, "time", lambda: real_time() + 120)
    assert verify_session_token(token) is None


@pytest.mark.parametrize("bad", ["", "abc", "a.b.c", "!!!.???", None, "x" * 5000])
def test_malformed_tokens_are_rejected(bad):
    assert verify_session_token(bad) is None


def test_revoked_token_is_rejected():
    token = create_session_token(USER)
    revoke_session_token(token)
    assert verify_session_token(token) is None


def test_ttl_comes_from_env(monkeypatch):
    monkeypatch.setenv("SESSION_TTL_DAYS", "2")
    token = create_session_token(USER)
    payload_b64 = token.split(".")[0]
    payload = json.loads(base64.urlsafe_b64decode(payload_b64 + "=" * (-len(payload_b64) % 4)))
    assert payload["exp"] - payload["iat"] == 2 * 86400


def test_missing_secret_falls_back_to_process_secret(monkeypatch):
    monkeypatch.delenv("SESSION_SECRET", raising=False)
    token = create_session_token(USER)
    assert verify_session_token(token) is not None  # stable within the same process


def test_restore_user_from_cookies():
    token = create_session_token(USER)
    assert restore_user_from_cookies({SESSION_COOKIE_NAME: token})["email"] == USER["email"]
    assert restore_user_from_cookies({}) is None
    assert restore_user_from_cookies({SESSION_COOKIE_NAME: "garbage"}) is None
    assert restore_user_from_cookies(None) is None


def test_set_cookie_script_contains_token_and_attributes():
    token = create_session_token(USER)
    script = build_set_cookie_script(token)
    assert script.startswith("<script>") and script.endswith("</script>")
    assert f"{SESSION_COOKIE_NAME}={token}" in script
    assert "path=/" in script and "SameSite=Lax" in script and "max-age=" in script
    assert "https:" in script  # adds Secure when served over HTTPS


def test_set_cookie_script_refuses_unsafe_values():
    with pytest.raises(ValueError):
        build_set_cookie_script("abc'; alert(1); '")


def test_clear_cookie_script_expires_cookie():
    script = build_clear_cookie_script()
    assert f"{SESSION_COOKIE_NAME}=" in script and "max-age=0" in script
