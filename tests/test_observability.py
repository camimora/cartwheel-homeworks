"""Homework 2, Part D: authentication tests for the session endpoints.

These tests run offline: no Langfuse, no Docker, no model provider key.

How to approach them:
  - Call the endpoint functions directly (server_app.create_session,
    server_app.post_message), as tests/test_hw_holes.py does for
    create_session. Avoid fastapi.testclient.TestClient: it runs the server
    lifespan, which loads .env and turns tracing on.
  - The `world` fixture (tests/conftest.py) seeds a temporary database and
    points the app at it. Request it as a test argument.
  - Clear server_app._SESSIONS at the start of each test so state from one
    test cannot leak into another.
  - Expect failures with:
        with pytest.raises(HTTPException) as exc:
            ...
        assert exc.value.status_code == 403
  - post_message is async; run it with asyncio.run(server_app.post_message(...)).
"""

from __future__ import annotations
import asyncio
import pytest
from fastapi import HTTPException
from server import app as server_app


def test_create_session_rejects_role_that_differs_from_database(world: dict) -> None:
    server_app._SESSIONS.clear()
    with pytest.raises(HTTPException) as exc:
        server_app.create_session(server_app.SessionCreate(user_id=1, role="merchant"))
    assert exc.value.status_code == 403
    assert server_app._SESSIONS == {}


def test_token_for_one_session_cannot_authorize_another(world: dict) -> None:
    server_app._SESSIONS.clear()
    session_shopper_A = server_app.create_session(server_app.SessionCreate(user_id=1, role="shopper"))
    session_merchant_B = server_app.create_session(server_app.SessionCreate(user_id=9001, role="merchant"))
    msg = server_app.MessageIn(message="My message")
    authorization=f"Bearer {session_shopper_A['token']}"
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            server_app.post_message(session_merchant_B["session_id"], msg, authorization=authorization)
        )
    assert exc.value.status_code == 403
    assert exc.value.detail == "token is for another session"

