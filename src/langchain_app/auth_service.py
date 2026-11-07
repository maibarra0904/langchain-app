"""Google OAuth and Session Authentication Service.

Complies strictly with Google's OAuth 2.0 Secure Response Handling Policy:
1. Always uses `response_type=code` with PKCE (RFC 7636, S256).
2. Avoids deprecated/blocked implicit flows (`response_type=token` or `id_token token`).
3. Supports both secretless PKCE for client apps and server flows when `GOOGLE_CLIENT_SECRET` is present.
4. Validates Google JWT ID tokens with Google public certs.
5. Automatically upserts authenticated users into Supabase `users` table.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import urllib.parse
from typing import Any, Optional
import requests
from dotenv import load_dotenv
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token
from langsmith import traceable

from langchain_app.supabase_service import get_supabase_client, is_supabase_configured, upsert_google_user

load_dotenv()
logger = logging.getLogger("langchain_app.auth_service")

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"


def is_google_auth_configured() -> bool:
    """Check if Google OAuth Client ID is configured in .env."""
    client_id = os.getenv("GOOGLE_CLIENT_ID", "").strip()
    placeholder_terms = {"tu_google_client_id", "your_client_id", "example.apps.googleusercontent.com"}
    if not client_id:
        return False
    if any(p in client_id.lower() for p in placeholder_terms):
        return False
    return True


def has_client_secret() -> bool:
    """Check if optional GOOGLE_CLIENT_SECRET is present."""
    secret = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()
    placeholder_terms = {"tu_google_client_secret", "your_client_secret"}
    if not secret or any(p in secret.lower() for p in placeholder_terms):
        return False
    return True


def generate_pkce_pair() -> tuple[str, str]:
    """Generate (code_verifier, code_challenge) adhering to RFC 7636 with SHA-256."""
    code_verifier = base64.urlsafe_b64encode(os.urandom(32)).decode("utf-8").rstrip("=")
    digest = hashlib.sha256(code_verifier.encode("utf-8")).digest()
    code_challenge = base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")
    return code_verifier, code_challenge


def get_google_auth_url(
    redirect_uri: str = "http://localhost:8501",
    state: str = "state",
    code_challenge: Optional[str] = None,
) -> str:
    """Generate Google OAuth 2.0 URL compliant with Google's Secure Response Handling policy.
    
    Always uses `response_type=code` (never implicit tokens).
    If code_challenge is provided, adds PKCE S256 parameters.
    """
    client_id = os.getenv("GOOGLE_CLIENT_ID", "").strip()
    if not client_id:
        return "#"

    params: dict[str, str] = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",  # Mandatory per Google policy!
        "scope": "openid email profile",
        "prompt": "select_account",
        "state": state,
    }

    if code_challenge:
        params["code_challenge"] = code_challenge
        params["code_challenge_method"] = "S256"

    return f"{GOOGLE_AUTH_URL}?{urllib.parse.urlencode(params)}"


def get_google_identity_button_html(client_id: str) -> str:
    """Generate official Google Identity Services (GIS) button HTML component.
    
    Uses Google's modern GSI SDK popup mode, capturing the signed ID token without URL redirects.
    """
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <script src="https://accounts.google.com/gsi/client" async defer></script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            background: transparent;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}
    </style>
</head>
<body>
    <div id="g_id_onload"
         data-client_id="{client_id}"
         data-context="signin"
         data-callback="handleCredentialResponse"
         data-auto_prompt="false">
    </div>
    <div class="g_id_signin"
         data-type="standard"
         data-shape="rectangular"
         data-theme="outline"
         data-text="signin_with"
         data-size="large"
         data-logo_alignment="left"
         data-width="300">
    </div>
    <script>
        function handleCredentialResponse(response) {{
            if (response && response.credential) {{
                window.parent.location.search = '?credential=' + encodeURIComponent(response.credential);
            }}
        }}
    </script>
</body>
</html>"""


def get_supabase_google_auth_url(redirect_uri: str = "http://localhost:8501") -> str:
    """Generate Google Auth URL through Supabase Auth (when Google provider is enabled in Supabase)."""
    client = get_supabase_client()
    if not client:
        return "#"
    try:
        res = client.auth.sign_in_with_oauth({
            "provider": "google",
            "options": {
                "redirect_to": redirect_uri,
            },
        })
        return getattr(res, "url", "#")
    except Exception as exc:
        logger.debug("Could not get Supabase OAuth URL: %s", exc)
        return "#"


@traceable(run_type="chain", name="verify_google_id_token", tags=["auth", "oauth", "jwt"])
def verify_and_process_id_token(token: str) -> Optional[dict[str, Any]]:
    """Verify a Google JWT ID Token directly using Google's public certs.
    
    NO client_secret required. Cryptographically verified against Google certs.
    """
    client_id = os.getenv("GOOGLE_CLIENT_ID", "").strip()
    try:
        req = google_requests.Request()
        id_info = google_id_token.verify_oauth2_token(
            token,
            req,
            audience=client_id if client_id else None,
        )

        google_id = id_info.get("sub", "")
        email = id_info.get("email", "")
        name = id_info.get("name", email.split("@")[0] if email else "User")
        picture = id_info.get("picture", "")

        db_user = upsert_google_user(
            google_id=google_id,
            email=email,
            full_name=name,
            avatar_url=picture,
        )

        return {
            "google_id": google_id,
            "email": email,
            "name": name,
            "avatar": picture,
            "db_id": db_user.get("id"),
            "authenticated": True,
            "auth_method": "google_id_token_verified",
        }
    except Exception as exc:
        logger.error("Failed to verify Google ID token: %s", exc)
        return None


@traceable(run_type="chain", name="exchange_google_code_for_user", tags=["auth", "oauth"])
def exchange_code_for_user_info(
    code: str,
    redirect_uri: str = "http://localhost:8501",
    code_verifier: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Exchange authorization code for user profile using PKCE or client_secret."""
    client_id = os.getenv("GOOGLE_CLIENT_ID", "").strip()
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()

    if not client_id:
        logger.error("Cannot authenticate: GOOGLE_CLIENT_ID missing.")
        return None

    # If code is actually an ID Token passed directly
    if code.count(".") == 2 and not client_secret:
        return verify_and_process_id_token(code)

    try:
        payload = {
            "code": code,
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        }
        if client_secret:
            payload["client_secret"] = client_secret
        if code_verifier:
            payload["code_verifier"] = code_verifier

        token_response = requests.post(
            GOOGLE_TOKEN_URL,
            data=payload,
            timeout=10,
        )
        token_data = token_response.json()
        access_token = token_data.get("access_token")
        id_tok = token_data.get("id_token")

        if id_tok:
            verified = verify_and_process_id_token(id_tok)
            if verified:
                return verified

        if not access_token:
            logger.error("Token exchange failed: %s", token_data)
            return None

        user_response = requests.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        profile = user_response.json()

        google_id = profile.get("sub", "")
        email = profile.get("email", "")
        name = profile.get("name", email.split("@")[0] if email else "User")
        picture = profile.get("picture", "")

        db_user = upsert_google_user(
            google_id=google_id,
            email=email,
            full_name=name,
            avatar_url=picture,
        )

        return {
            "google_id": google_id,
            "email": email,
            "name": name,
            "avatar": picture,
            "db_id": db_user.get("id"),
            "authenticated": True,
            "auth_method": "google_oauth_code_flow",
        }
    except Exception as exc:
        logger.error("Failed to authenticate user via Google: %s", exc)
        return None


def get_mock_google_user(email: str = "usuario.demo@gmail.com", name: str = "Usuario Demo") -> dict[str, Any]:
    """Provide a mock Google user profile for local offline testing."""
    google_id = "demo_google_id_1029384756"
    avatar = "https://lh3.googleusercontent.com/a/default-user=s96-c"
    db_user = upsert_google_user(
        google_id=google_id,
        email=email,
        full_name=name,
        avatar_url=avatar,
    )
    return {
        "google_id": google_id,
        "email": email,
        "name": name,
        "avatar": avatar,
        "db_id": db_user.get("id", f"mock-{google_id[:8]}"),
        "authenticated": True,
        "auth_method": "local_mock",
    }
