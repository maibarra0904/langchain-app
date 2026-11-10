"""Persistent login sessions for the Streamlit app.

Streamlit keeps ``st.session_state`` per browser connection, so a page reload starts a
brand-new session and the logged-in user is lost. This service persists the login in a
browser cookie holding a **signed, expiring token**:

- ``create_session_token``: ``base64url(payload).base64url(HMAC-SHA256(SESSION_SECRET, payload))``
  with the user's public profile (no passwords or OAuth tokens), ``iat``, ``exp`` and a ``jti``.
- ``verify_session_token``: constant-time signature check + expiry + revocation.
- The cookie is written/cleared from the browser with a tiny inline script
  (``st.html(..., unsafe_allow_javascript=True)``) and read back on reload through
  ``st.context.cookies`` (cookies sent with the initial request of each session).

Tampering with the cookie invalidates the signature, so the identity in it can be trusted.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import threading
import time
from typing import Any, Mapping, Optional

logger = logging.getLogger("langchain_app.session_service")

SESSION_COOKIE_NAME = "lagchain_session"
DEFAULT_TTL_DAYS = 7
MAX_TOKEN_LENGTH = 3500  # browsers cap a cookie at ~4 KB
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$")
USER_FIELDS = ("google_id", "email", "name", "avatar", "db_id", "auth_method")

_process_secret: Optional[bytes] = None
_secret_lock = threading.Lock()
_revoked_jti: set[str] = set()


def _get_secret() -> bytes:
    """Signing key from SESSION_SECRET; falls back to a random per-process key (with a warning)."""
    configured = os.getenv("SESSION_SECRET", "").strip()
    if configured:
        return configured.encode("utf-8")
    global _process_secret
    with _secret_lock:
        if _process_secret is None:
            logger.warning(
                "SESSION_SECRET is not set: using a random per-process key. Logins survive page "
                "reloads but not server restarts. Define SESSION_SECRET in .env."
            )
            _process_secret = secrets.token_bytes(32)
        return _process_secret


def get_session_ttl_seconds() -> int:
    try:
        days = float(os.getenv("SESSION_TTL_DAYS", str(DEFAULT_TTL_DAYS)))
    except ValueError:
        days = DEFAULT_TTL_DAYS
    return max(60, int(days * 86400))


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64decode(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def _sign(payload_b64: str) -> str:
    return _b64encode(hmac.new(_get_secret(), payload_b64.encode("ascii"), hashlib.sha256).digest())


def create_session_token(user: Mapping[str, Any], ttl_seconds: Optional[int] = None) -> str:
    """Create a signed session token carrying the user's public profile."""
    now = int(time.time())
    payload = {field: user.get(field) for field in USER_FIELDS if user.get(field) is not None}
    payload.update({
        "iat": now,
        "exp": now + (ttl_seconds if ttl_seconds is not None else get_session_ttl_seconds()),
        "jti": secrets.token_urlsafe(12),
    })
    payload_b64 = _b64encode(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    return f"{payload_b64}.{_sign(payload_b64)}"


def _decode_verified(token: Any) -> Optional[dict[str, Any]]:
    if not isinstance(token, str) or len(token) > MAX_TOKEN_LENGTH or not _TOKEN_RE.match(token):
        return None
    payload_b64, signature = token.split(".")
    if not hmac.compare_digest(_sign(payload_b64), signature):
        return None
    try:
        payload = json.loads(_b64decode(payload_b64))
    except (ValueError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def verify_session_token(token: Any) -> Optional[dict[str, Any]]:
    """Return the user stored in a valid token, or None if invalid, expired or revoked."""
    payload = _decode_verified(token)
    if not payload:
        return None
    if int(payload.get("exp", 0)) <= time.time():
        return None
    if payload.get("jti") in _revoked_jti:
        return None
    if not payload.get("email") and not payload.get("google_id"):
        return None

    user = {field: payload.get(field) for field in USER_FIELDS}
    user["authenticated"] = True
    user["restored_from_cookie"] = True
    return user


def revoke_session_token(token: Any) -> None:
    """Invalidate a token server-side (logout), even if the browser still holds the cookie."""
    payload = _decode_verified(token)
    if payload and payload.get("jti"):
        _revoked_jti.add(str(payload["jti"]))


def restore_user_from_cookies(cookies: Optional[Mapping[str, str]]) -> Optional[dict[str, Any]]:
    """Restore the logged-in user from the request cookies (``st.context.cookies``)."""
    if not cookies:
        return None
    return verify_session_token(cookies.get(SESSION_COOKIE_NAME))


def build_set_cookie_script(token: str, max_age: Optional[int] = None) -> str:
    """Inline script that stores the session cookie in the browser (adds Secure over HTTPS)."""
    if not isinstance(token, str) or not _TOKEN_RE.match(token):
        raise ValueError("Refusing to write an unexpected value into the session cookie.")
    age = max_age if max_age is not None else get_session_ttl_seconds()
    return (
        "<script>"
        f"document.cookie = '{SESSION_COOKIE_NAME}={token}; path=/; max-age={int(age)}; SameSite=Lax'"
        " + (window.location.protocol === 'https:' ? '; Secure' : '');"
        "</script>"
    )


def build_clear_cookie_script() -> str:
    """Inline script that deletes the session cookie (logout)."""
    return f"<script>document.cookie = '{SESSION_COOKIE_NAME}=; path=/; max-age=0; SameSite=Lax';</script>"
