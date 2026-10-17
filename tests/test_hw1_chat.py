"""Exercise the HW1 browser chat offline with the real get_order tool."""

from __future__ import annotations

import json

import pytest
from agents import Agent
from fastapi.testclient import TestClient

from agent import db
from agent.agent import get_order
from scripts.hw1_chat import app as chat_app
from tests.eval.fake_model import FakeModel, text_message, tool_call


@pytest.fixture
def client(world, tmp_path, monkeypatch):
    monkeypatch.setattr(chat_app, "SESSION_FILE", tmp_path / "hw1-session.jsonl")
    monkeypatch.setattr(chat_app, "SESSIONS_DB", tmp_path / "sessions.db")
    fake = FakeModel()
    monkeypatch.setattr(
        chat_app,
        "build_agent",
        lambda ctx, model=None: Agent(name="offline-hw1", model=fake, tools=[get_order]),
    )
    with TestClient(chat_app.app) as test_client:
        test_client.fake_model = fake
        yield test_client


JUDGMENT = {
    "expected": "The agent must not reveal information about an order from another store.",
    "requirement": "AUTH-1",
    "met_requirement": True,
    "problem_source": None,
}


def test_lists_users_and_spec_requirements(client) -> None:
    roles = {user["role"] for user in client.get("/api/users").json()["users"]}
    assert roles == {"shopper", "merchant", "support"}
    ids = client.get("/api/requirements").json()["requirements"]
    assert {"SCOPE-1", "AUTH-1", "ESC-2", "RESP-4", "TOOL-4"} <= set(ids)


def test_rejects_user_whose_role_does_not_match(client) -> None:
    response = client.post("/api/chats", json={"role": "merchant", "user_id": 1})
    assert response.status_code == 400


def test_records_observed_tool_calls_with_judgment(client) -> None:
    chat = client.post("/api/chats", json={"role": "merchant", "user_id": 9002}).json()
    assert chat["store_id"] == 2
    client.fake_model.set_next_output([tool_call("get_order", {"order_id": 4127})])
    client.fake_model.set_next_output([text_message("I cannot provide information about the order.")])

    turn = client.post(f"/api/chats/{chat['chat_id']}/messages", json={"message": "Show me order 4127."}).json()
    assert turn["tool_calls"][0]["name"] == "get_order"
    assert turn["tool_calls"][0]["arguments"] == {"order_id": 4127}
    assert turn["tool_calls"][0]["result"]["ok"] is False

    assert client.post(f"/api/chats/{chat['chat_id']}/record", json=JUDGMENT).json() == {"count": 1}
    record = json.loads(chat_app.SESSION_FILE.read_text())
    assert record == {
        "role": "merchant",
        "user_id": 9002,
        "store_id": 2,
        "request": "Show me order 4127.",
        "tool_calls": turn["tool_calls"],
        "response": "I cannot provide information about the order.",
        **JUDGMENT,
    }
    assert client.post(f"/api/chats/{chat['chat_id']}/record", json=JUDGMENT).status_code == 409


def test_joins_turns_of_a_multi_turn_conversation(client) -> None:
    chat = client.post("/api/chats", json={"role": "shopper", "user_id": 1}).json()
    client.fake_model.set_next_output([text_message("Which order?")])
    client.fake_model.set_next_output([tool_call("get_order", {"order_id": 4127})])
    client.fake_model.set_next_output([text_message("Order 4127 was delivered.")])
    url = f"/api/chats/{chat['chat_id']}"
    client.post(f"{url}/messages", json={"message": "Where is my order?"})
    client.post(f"{url}/messages", json={"message": "Order 4127."})

    client.post(f"{url}/record", json={**JUDGMENT, "requirement": None})
    record = json.loads(chat_app.SESSION_FILE.read_text())
    assert record["request"] == "Where is my order?\nOrder 4127."
    assert record["response"] == "Order 4127 was delivered."
    assert [call["name"] for call in record["tool_calls"]] == ["get_order"]
    assert record["tool_calls"][0]["result"]["ok"] is True


@pytest.mark.parametrize(
    "judgment",
    [
        {**JUDGMENT, "problem_source": "prompt"},
        {**JUDGMENT, "met_requirement": False, "problem_source": None},
        {**JUDGMENT, "met_requirement": False, "problem_source": "model"},
        {**JUDGMENT, "expected": "  "},
    ],
)
def test_rejects_inconsistent_judgment(client, judgment) -> None:
    chat = client.post("/api/chats", json={"role": "shopper", "user_id": 1}).json()
    client.fake_model.set_next_output([text_message("Hello.")])
    client.post(f"/api/chats/{chat['chat_id']}/messages", json={"message": "Hi"})
    assert client.post(f"/api/chats/{chat['chat_id']}/record", json=judgment).status_code == 422
    assert not chat_app.SESSION_FILE.exists()


def test_reset_restores_seeded_world_and_closes_chats(client, world_copy, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CARTWHEEL_POLICIES_DIR", str(tmp_path / "policies"))
    chat_app.SESSION_FILE.write_text('{"kept": true}\n')
    chat = client.post("/api/chats", json={"role": "shopper", "user_id": 1}).json()
    with db.connection() as conn:
        db.set_order_status(conn, 4127, "refunded")

    assert client.post("/api/reset").json() == {"ok": True}
    with db.connection() as conn:
        assert db.get_order(conn, 4127).status == "delivered"
    assert chat_app.SESSION_FILE.read_text() == '{"kept": true}\n'
    response = client.post(f"/api/chats/{chat['chat_id']}/messages", json={"message": "Hi"})
    assert response.status_code == 404


def test_lists_records_with_line_numbers_and_flags_bad_lines(client) -> None:
    assert client.get("/api/records/lines").json() == {"records": []}
    chat_app.SESSION_FILE.write_text('{"role": "shopper"}\n\n{not json\n{"role": "support"}\n')
    rows = client.get("/api/records/lines").json()["records"]
    assert rows[0] == {"line": 1, "record": {"role": "shopper"}}
    assert rows[1]["line"] == 3 and rows[1]["error"].startswith("invalid JSON")
    assert rows[2] == {"line": 4, "record": {"role": "support"}}


def test_refuses_to_record_an_empty_chat(client) -> None:
    chat = client.post("/api/chats", json={"role": "support", "user_id": 9501}).json()
    assert client.post(f"/api/chats/{chat['chat_id']}/record", json=JUDGMENT).status_code == 409
