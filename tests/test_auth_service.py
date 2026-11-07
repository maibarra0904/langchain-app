"""Unit tests for auth_service module."""

import os
import urllib.parse
from unittest.mock import MagicMock, patch

import pytest

from langchain_app.auth_service import (
    exchange_code_for_user_info,
    generate_pkce_pair,
    get_google_auth_url,
    get_google_identity_button_html,
    has_client_secret,
    is_google_auth_configured,
    verify_and_process_id_token,
)


def test_is_google_auth_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)
    assert not is_google_auth_configured()

    monkeypatch.setenv("GOOGLE_CLIENT_ID", "tu_google_client_id")
    assert not is_google_auth_configured()

    monkeypatch.setenv("GOOGLE_CLIENT_ID", "12345-abc.apps.googleusercontent.com")
    assert is_google_auth_configured()


def test_has_client_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_CLIENT_SECRET", raising=False)
    assert not has_client_secret()

    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "tu_google_client_secret")
    assert not has_client_secret()

    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "GOCSPX-real-secret-12345")
    assert has_client_secret()


def test_generate_pkce_pair() -> None:
    verifier, challenge = generate_pkce_pair()
    assert isinstance(verifier, str) and len(verifier) > 20
    assert isinstance(challenge, str) and len(challenge) > 20
    assert verifier != challenge


def test_get_google_auth_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client-id.apps.googleusercontent.com")
    monkeypatch.setenv("GOOGLE_REDIRECT_URI", "http://localhost:8501")

    url = get_google_auth_url()
    assert url.startswith("https://accounts.google.com/o/oauth2/v2/auth?")
    parsed = urllib.parse.urlparse(url)
    qs = urllib.parse.parse_qs(parsed.query)

    assert qs["client_id"] == ["test-client-id.apps.googleusercontent.com"]
    assert qs["redirect_uri"] == ["http://localhost:8501"]
    assert qs["response_type"] == ["code"]
    assert qs["scope"] == ["openid email profile"]
    assert "code_challenge" not in qs

    # With PKCE challenge
    url_pkce = get_google_auth_url(code_challenge="my-challenge")
    parsed_pkce = urllib.parse.urlparse(url_pkce)
    qs_pkce = urllib.parse.parse_qs(parsed_pkce.query)
    assert qs_pkce["code_challenge"] == ["my-challenge"]
    assert qs_pkce["code_challenge_method"] == ["S256"]


def test_get_google_identity_button_html() -> None:
    html = get_google_identity_button_html("test-id-123")
    assert "test-id-123" in html
    assert "accounts.google.com/gsi/client" in html
    assert "handleCredentialResponse" in html


@patch("langchain_app.auth_service.google_id_token.verify_oauth2_token")
@patch("langchain_app.auth_service.upsert_google_user")
def test_verify_and_process_id_token_success(
    mock_upsert: MagicMock,
    mock_verify: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client-id")
    mock_verify.return_value = {
        "sub": "google-123",
        "email": "tester@example.com",
        "name": "Test User",
        "picture": "https://example.com/avatar.jpg",
    }
    mock_upsert.return_value = {"id": "db-user-uuid-1"}

    result = verify_and_process_id_token("fake.jwt.token")
    assert result is not None
    assert result["google_id"] == "google-123"
    assert result["email"] == "tester@example.com"
    assert result["name"] == "Test User"
    assert result["avatar"] == "https://example.com/avatar.jpg"
    assert result["db_id"] == "db-user-uuid-1"
    assert result["authenticated"] is True


@patch("langchain_app.auth_service.google_id_token.verify_oauth2_token")
def test_verify_and_process_id_token_failure(mock_verify: MagicMock) -> None:
    mock_verify.side_effect = Exception("Invalid signature")
    result = verify_and_process_id_token("invalid.token")
    assert result is None


@patch("langchain_app.auth_service.requests.post")
@patch("langchain_app.auth_service.verify_and_process_id_token")
def test_exchange_code_for_user_info_with_client_secret(
    mock_verify: MagicMock,
    mock_post: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "test-secret")
    monkeypatch.setenv("GOOGLE_REDIRECT_URI", "http://localhost:8501")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "access_token": "ya29.fake-access-token",
        "id_token": "header.payload.signature",
    }
    mock_post.return_value = mock_resp

    mock_verify.return_value = {
        "google_id": "google-456",
        "email": "code_user@example.com",
        "name": "Code User",
        "authenticated": True,
    }

    res = exchange_code_for_user_info("auth_code_123")
    assert res is not None
    assert res["email"] == "code_user@example.com"
    mock_post.assert_called_once()

