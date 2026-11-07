"""Google OAuth and Session Authentication Service.

Supports:
1. Google OAuth 2.0 Authorization Code flow for Streamlit applications.
2. Supabase Auth OAuth provider generation when Supabase is enabled.
3. Automatic user upsert into Supabase `users` table.
4. Seamless fallback for local developer testing.
"""

from __future__ import annotations

import logging
import os
import urllib.parse
from typing import Any, Optional
import requests
from dotenv import load_dotenv
from langsmith import traceable

from langchain_app.supabase_service import upsert_google_user

load_dotenv()
logger = logging.getLogger("langchain_app.auth_service")

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"


def is_google_auth_configured() -> bool:
    """Check if Google OAuth client ID and secret are configured in .env."""
    client_id = os.getenv("GOOGLE_CLIENT_ID", "").strip()
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()
    placeholder_terms = {"tu_google_client_id", "your_client_id", "example.apps.googleusercontent.com"}
    if not client_id or not client_secret:
        return False
    if any(p in client_id.lower() for p in placeholder_terms):
        return False
    return True


def get_google_auth_url(redirect_uri: str = "http://localhost:8501", state: str = "state") -> str:
    """Generate the Google OAuth 2.0 authorization URL for login."""
    client_id = os.getenv("GOOGLE_CLIENT_ID", "").strip()
    if not client_id:
        return "#"

    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "select_account",
        "state": state,
    }
    return f"{GOOGLE_AUTH_URL}?{urllib.parse.urlencode(params)}"


@traceable(run_type="chain", name="exchange_google_code_for_user", tags=["auth", "oauth"])
def exchange_code_for_user_info(code: str, redirect_uri: str = "http://localhost:8501") -> Optional[dict[str, Any]]:
    """Exchange authorization code for user profile from Google.
    
    Args:
        code: Authorization code from Google redirect query param.
        redirect_uri: Redirect URI registered in Google Console.
        
    Returns:
        User profile dict or None.
    """
    client_id = os.getenv("GOOGLE_CLIENT_ID", "").strip()
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()

    if not client_id or not client_secret:
        logger.error("Cannot exchange code: Google OAuth credentials missing.")
        return None

    try:
        # 1. Exchange code for access token
        token_response = requests.post(
            GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
            timeout=10,
        )
        token_data = token_response.json()
        access_token = token_data.get("access_token")

        if not access_token:
            logger.error("Token exchange failed: %s", token_data)
            return None

        # 2. Fetch user profile with access token
        user_response = requests.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        profile = user_response.json()

        # 3. Synchronize user with Supabase database
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
    }
