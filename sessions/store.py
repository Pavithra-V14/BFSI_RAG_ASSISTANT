from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import config

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "app.db"
DB_PATH.parent.mkdir(exist_ok=True)

_USE_POSTGRES = config.DATABASE_URL is not None
_PG_ADVISORY_LOCK_ID = 918_273_647  
def _translate_placeholders(sql: str) -> str:
    return sql.replace("%s", "?")

@contextmanager
def _conn():
    if _USE_POSTGRES:
        import psycopg2
        import psycopg2.extras
        conn = psycopg2.connect(config.DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def _execute(conn, sql: str, params: tuple = ()):
    cur = conn.cursor()
    cur.execute(sql if _USE_POSTGRES else _translate_placeholders(sql), params)
    return cur

def init_db() -> None:
    with _conn() as conn:
        if _USE_POSTGRES:
            _execute(conn, "SELECT pg_advisory_xact_lock(%s)", (_PG_ADVISORY_LOCK_ID,))
            _execute(conn, """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                )
            """)
            _execute(conn, """
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    citations TEXT,
                    grounded INTEGER,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)
        else:
            _execute(conn, """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            _execute(conn, """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    citations TEXT,
                    grounded INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)


@dataclass
class ChatSession:
    session_id: str
    user_id: int
    title: str
    created_at: str


def create_session(user_id: int, title: str = "New chat") -> ChatSession:
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        _execute(
            conn, "INSERT INTO sessions (session_id, user_id, title, created_at) VALUES (%s, %s, %s, %s)",
            (session_id, user_id, title, now),
        )
    return ChatSession(session_id=session_id, user_id=user_id, title=title, created_at=now)


def _normalize_row(row: dict) -> dict:
    """
    Postgres returns TIMESTAMPTZ columns as real datetime objects, SQLite
    returns plain strings, and every Pydantic response model in app.py
    expects created_at as a str. Confirmed live (real ValidationError on
    /sessions and /admin/users) the moment this ran against Postgres for
    the first time — worked fine against SQLite the whole time before
    that, which is exactly why this class of bug is easy to miss without
    testing against BOTH backends specifically.
    """
    d = dict(row)
    if "created_at" in d and hasattr(d["created_at"], "isoformat"):
        d["created_at"] = d["created_at"].isoformat()
    return d


def list_sessions(user_id: int) -> list[dict]:
    with _conn() as conn:
        cur = _execute(conn, "SELECT * FROM sessions WHERE user_id = %s ORDER BY created_at DESC", (user_id,))
        return [_normalize_row(r) for r in cur.fetchall()]


def get_session(session_id: str, user_id: int) -> dict | None:
    """Ownership check baked into the query — not a separate step."""
    with _conn() as conn:
        cur = _execute(conn, "SELECT * FROM sessions WHERE session_id = %s AND user_id = %s", (session_id, user_id))
        row = cur.fetchone()
        return _normalize_row(row) if row else None


def rename_session_if_first_message(session_id: str, user_id: int, first_query: str) -> None:
    session = get_session(session_id, user_id)
    if session and session["title"] == "New chat":
        title = (first_query[:50] + "…") if len(first_query) > 50 else first_query
        with _conn() as conn:
            _execute(conn, "UPDATE sessions SET title = %s WHERE session_id = %s", (title, session_id))


def delete_session(session_id: str, user_id: int) -> bool:
    session = get_session(session_id, user_id)
    if not session:
        return False
    with _conn() as conn:
        _execute(conn, "DELETE FROM messages WHERE session_id = %s", (session_id,))
        _execute(conn, "DELETE FROM sessions WHERE session_id = %s", (session_id,))
    return True


def add_message(session_id: str, role: str, content: str, citations: list[str] | None = None, grounded: bool | None = None) -> None:
    with _conn() as conn:
        _execute(
            conn, "INSERT INTO messages (session_id, role, content, citations, grounded) VALUES (%s, %s, %s, %s, %s)",
            (session_id, role, content, json.dumps(citations or []), int(grounded) if grounded is not None else None),
        )


def get_messages(session_id: str, user_id: int) -> list[dict]:
    if not get_session(session_id, user_id):
        return []
    with _conn() as conn:
        cur = _execute(conn, "SELECT * FROM messages WHERE session_id = %s ORDER BY id ASC", (session_id,))
        out = []
        for r in cur.fetchall():
            d = _normalize_row(r)
            d["citations"] = json.loads(d["citations"] or "[]")
            d["grounded"] = bool(d["grounded"]) if d["grounded"] is not None else None
            out.append(d)
        return out


init_db()