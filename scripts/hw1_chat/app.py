"""Local browser chat for the Homework 1 manual session.

A student convenience, separate from ``server/app.py`` (whose session routes
are Homework 2). Each chat builds the agent exactly as ``agent.cli`` does:
the auth context comes from the users table through ``cli.resolve_auth``,
never from the chat. The Record button appends one line to
``hw1-session.jsonl`` with the observed request, tool calls, and response,
plus the judgment fields the student typed into the form. The Records tab
shows the file read-only, and the DB tab browses the world database through
a read-only SQLite connection. Reset DB
regenerates the seeded world, like ``uv run python -m seed.generate``.

Run from the repository root:
    uv run uvicorn scripts.hw1_chat.app:app --port 8020
then open http://127.0.0.1:8020.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from contextlib import asynccontextmanager, closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

from agents import Agent, RunConfig, Runner, SQLiteSession
from agents.items import RunItem
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agent import db
from agent.agent import build_agent, prompt_version
from agent.auth import AuthContext
from agent.cli import MAX_TURNS, SESSIONS_DB, resolve_auth
from agent import config
from agent.config import REPO_ROOT
from observability.instrument import load_env
from seed.generate import generate_world

SESSION_FILE = REPO_ROOT / "hw1-session.jsonl"
SPEC_FILE = REPO_ROOT / "SPEC.md"
INDEX_HTML = Path(__file__).resolve().parent / "index.html"
PROBLEM_SOURCES = {"prompt", "tool", "specification"}
DB_PAGE_SIZE = 50


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    load_env()
    yield


app = FastAPI(title="Cartwheel HW1 chat", lifespan=lifespan)
_write_lock = threading.Lock()


@dataclass
class Chat:
    ctx: AuthContext
    agent: Agent[AuthContext]
    session: SQLiteSession
    turns: list[dict[str, Any]] = field(default_factory=list)
    recorded: bool = False


_chats: dict[str, Chat] = {}


class NewChat(BaseModel):
    role: str
    user_id: int
    model: str | None = None


class Message(BaseModel):
    message: str


class Judgment(BaseModel):
    expected: str
    requirement: str | None
    met_requirement: bool
    problem_source: str | None


def tool_calls_from(new_items: list[RunItem]) -> list[dict[str, Any]]:
    """Pair each tool call with its result, mirroring ``cli._print_tool_calls``."""
    outputs = {
        item.call_id: item.output
        for item in new_items
        if item.type == "tool_call_output_item" and item.call_id is not None
    }
    calls = []
    for item in new_items:
        if item.type != "tool_call_item":
            continue
        raw = item.raw_item
        args = raw.get("arguments") if isinstance(raw, dict) else getattr(raw, "arguments", None)
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        calls.append({
            "name": item.tool_name,
            "arguments": args,
            "result": outputs.get(item.call_id),
        })
    return calls


def build_record(chat: Chat, judgment: Judgment) -> dict[str, Any]:
    """Join every turn: all requests in order, every tool call, the final reply."""
    return {
        "role": chat.ctx.role,
        "user_id": chat.ctx.user_id,
        "store_id": chat.ctx.store_id,
        "request": "\n".join(turn["request"] for turn in chat.turns),
        "tool_calls": [call for turn in chat.turns for call in turn["tool_calls"]],
        "response": chat.turns[-1]["response"],
        "expected": judgment.expected,
        "requirement": judgment.requirement,
        "met_requirement": judgment.met_requirement,
        "problem_source": judgment.problem_source,
    }


def validate_judgment(judgment: Judgment) -> None:
    if not judgment.expected.strip():
        raise HTTPException(422, "expected must describe the specified behavior")
    if judgment.met_requirement and judgment.problem_source is not None:
        raise HTTPException(422, "problem_source must be null when the requirement was met")
    if not judgment.met_requirement and judgment.problem_source not in PROBLEM_SOURCES:
        raise HTTPException(422, "problem_source must be prompt, tool, or specification")


def count_records() -> int:
    if not SESSION_FILE.exists():
        return 0
    return sum(1 for line in SESSION_FILE.read_text().splitlines() if line.strip())


def _get_chat(chat_id: str) -> Chat:
    chat = _chats.get(chat_id)
    if chat is None:
        raise HTTPException(404, "unknown chat; start a new one")
    return chat


@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.get("/api/users")
def users() -> dict[str, Any]:
    with db.connection() as conn:
        rows = conn.execute("SELECT id, name, role, store_id FROM users ORDER BY id").fetchall()
    return {"users": [dict(row) for row in rows]}


@app.get("/api/requirements")
def requirements() -> dict[str, Any]:
    text = SPEC_FILE.read_text()
    ids = re.findall(r"\*\*([A-Z]+-\d+)\.\*\*", text) + re.findall(
        r"^\|\s*([A-Z]+-\d+)\s*\|", text, flags=re.MULTILINE
    )
    return {"requirements": list(dict.fromkeys(ids))}


@app.get("/api/records")
def records() -> dict[str, Any]:
    return {"count": count_records(), "path": SESSION_FILE.name}


@app.get("/api/records/lines")
def record_lines() -> dict[str, Any]:
    """Every record with its 1-based line number, for the read-only Records tab.

    A line that is not valid JSON is returned with an error instead of
    failing the whole listing, so a hand edit gone wrong stays visible.
    """
    if not SESSION_FILE.exists():
        return {"records": []}
    rows = []
    for number, line in enumerate(SESSION_FILE.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append({"line": number, "record": json.loads(line)})
        except json.JSONDecodeError as exc:
            rows.append({"line": number, "error": f"invalid JSON: {exc.msg}"})
    return {"records": rows}


def _read_only_db() -> sqlite3.Connection:
    """Open the world database read-only, so the DB tab cannot write to it."""
    path = config.db_path()
    if not path.exists():
        raise HTTPException(404, f"{path.name} does not exist; reset the database")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _table_columns(conn: sqlite3.Connection) -> dict[str, list[str]]:
    names = [
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    return {
        name: [col["name"] for col in conn.execute(f'PRAGMA table_info("{name}")')]
        for name in names
    }


@app.get("/api/db/tables")
def db_tables() -> dict[str, Any]:
    with closing(_read_only_db()) as conn:
        tables = _table_columns(conn)
        return {
            "tables": [
                {
                    "name": name,
                    "columns": columns,
                    "rows": conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0],
                }
                for name, columns in tables.items()
            ]
        }


@app.get("/api/db/tables/{table}")
def db_rows(
    table: str, page: int = 1, column: str | None = None, value: str | None = None
) -> dict[str, Any]:
    """One page of a table, newest rows first, optionally filtered by column = value.

    Table and column names are checked against the schema before they are
    quoted into SQL; the filter value is always a bound parameter.
    """
    with closing(_read_only_db()) as conn:
        tables = _table_columns(conn)
        if table not in tables:
            raise HTTPException(404, f"unknown table {table!r}")
        where, params = "", []
        if column and value is not None and value != "":
            if column not in tables[table]:
                raise HTTPException(422, f"unknown column {column!r} in {table}")
            where, params = f'WHERE "{column}" = ?', [value]
        total = conn.execute(f'SELECT COUNT(*) FROM "{table}" {where}', params).fetchone()[0]
        page = max(page, 1)
        rows = conn.execute(
            f'SELECT * FROM "{table}" {where} ORDER BY rowid DESC LIMIT ? OFFSET ?',
            [*params, DB_PAGE_SIZE, (page - 1) * DB_PAGE_SIZE],
        ).fetchall()
        return {
            "table": table,
            "columns": tables[table],
            "rows": [list(row) for row in rows],
            "total": total,
            "page": page,
            "page_size": DB_PAGE_SIZE,
        }


@app.post("/api/reset")
def reset() -> dict[str, Any]:
    """Regenerate the seeded world, as ``uv run python -m seed.generate`` does.

    Open chats are closed so no recorded conversation spans two database
    states. ``hw1-session.jsonl`` and ``.sessions.db`` are not touched.
    """
    _chats.clear()
    generate_world(db_path=config.db_path(), policies_dir=config.policies_dir())
    return {"ok": True}


@app.post("/api/chats")
def new_chat(body: NewChat) -> dict[str, Any]:
    try:
        ctx = resolve_auth(body.role, body.user_id)
    except SystemExit as exc:
        raise HTTPException(400, str(exc)) from None
    chat_id = uuid.uuid4().hex[:12]
    _chats[chat_id] = Chat(
        ctx=ctx,
        agent=build_agent(ctx, model=body.model or None),
        session=SQLiteSession(
            f"hw1ui-{ctx.role}-{ctx.user_id}-{int(time.time())}", str(SESSIONS_DB)
        ),
    )
    return {
        "chat_id": chat_id,
        "role": ctx.role,
        "user_id": ctx.user_id,
        "store_id": ctx.store_id,
        "prompt_version": prompt_version(),
    }


@app.post("/api/chats/{chat_id}/messages")
async def send_message(chat_id: str, body: Message) -> dict[str, Any]:
    chat = _get_chat(chat_id)
    if chat.recorded:
        raise HTTPException(409, "this chat is already recorded; start a new one")
    if not body.message.strip():
        raise HTTPException(422, "message is empty")
    result = await Runner.run(
        chat.agent,
        body.message,
        session=chat.session,
        context=chat.ctx,
        max_turns=MAX_TURNS,
        run_config=RunConfig(tracing_disabled=True),
    )
    if getattr(result, "interruptions", None):
        raise HTTPException(500, "the run paused for approval; this chat does not handle pauses")
    turn = {
        "request": body.message,
        "tool_calls": tool_calls_from(result.new_items),
        "response": str(result.final_output),
    }
    chat.turns.append(turn)
    return turn


@app.post("/api/chats/{chat_id}/record")
def record(chat_id: str, judgment: Judgment) -> dict[str, Any]:
    chat = _get_chat(chat_id)
    if chat.recorded:
        raise HTTPException(409, "this chat is already recorded")
    if not chat.turns:
        raise HTTPException(409, "nothing to record yet")
    validate_judgment(judgment)
    line = json.dumps(build_record(chat, judgment), ensure_ascii=False, default=str)
    with _write_lock, SESSION_FILE.open("a") as handle:
        handle.write(line + "\n")
    chat.recorded = True
    return {"count": count_records()}
