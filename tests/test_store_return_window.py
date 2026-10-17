"""The student-added get_store_return_window tool (Homework 1, Part A)."""

from __future__ import annotations

import pytest

from agent import agent as agent_module
from agent.auth import AuthContext
from agent.tools import get_store_return_window

SHOPPER_1 = AuthContext(user_id=1, role="shopper")


def test_store_override_takes_precedence(world: dict) -> None:
    assert get_store_return_window(SHOPPER_1, "Juniper Home Goods") == {
        "ok": True,
        "store_id": 2,
        "store_name": "Juniper Home Goods",
        "return_window_days": 14,
        "has_override": True,
        "platform_return_window_days": 30,
        "policy_id": "store-juniper-home-goods-policy",
    }


def test_store_without_override_or_doc_uses_platform_window(world: dict) -> None:
    result = get_store_return_window(SHOPPER_1, "Blue Heron Ceramics")
    assert result["return_window_days"] == 30
    assert result["has_override"] is False
    assert result["policy_id"] is None


def test_store_with_doc_but_no_override_cites_its_doc(world: dict) -> None:
    result = get_store_return_window(SHOPPER_1, "Cascade Audio")
    assert result["return_window_days"] == 30
    assert result["has_override"] is False
    assert result["policy_id"] == "store-cascade-audio-policy"


@pytest.mark.parametrize("name", ["northwind books", "NORTHWIND BOOKS", "northwind-books", "  Northwind Books "])
def test_matches_name_or_slug_case_insensitively(world: dict, name: str) -> None:
    result = get_store_return_window(SHOPPER_1, name)
    assert (result["store_name"], result["return_window_days"]) == ("Northwind Books", 45)


def test_rejects_empty_name(world: dict) -> None:
    assert get_store_return_window(SHOPPER_1, "   ")["error"] == "invalid_argument"


def test_unknown_store_is_not_found(world: dict) -> None:
    result = get_store_return_window(SHOPPER_1, "Best Buy")
    assert result["ok"] is False
    assert result["error"] == "not_found"


@pytest.mark.parametrize("role", ["shopper", "merchant", "support"])
def test_registered_for_every_role(role: str) -> None:
    assert agent_module.get_store_return_window in agent_module.TOOLS_BY_ROLE[role]
