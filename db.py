"""
SQLite persistence layer.

Kept deliberately simple: one file, no ORM. For a hackathon demo this is
plenty, and it's trivial to read/audit in five minutes.
"""
import sqlite3
import os
import json
import time
from contextlib import contextmanager

DB_PATH = os.getenv("DB_PATH", "./atlas.db")


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                telegram_id     TEXT PRIMARY KEY,
                name            TEXT,
                onboarded       INTEGER DEFAULT 0,
                role            TEXT,           -- investor / analyst / founder / student / etc
                watchlist       TEXT DEFAULT '[]',  -- JSON list of tickers/companies
                interests       TEXT DEFAULT '[]',  -- JSON list of sectors/topics
                briefing_time   TEXT,           -- e.g. "08:00" (24h, user's local as they state it)
                notes           TEXT DEFAULT '',    -- free-form memory the assistant accumulates
                created_at      REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                telegram_id     TEXT,
                role            TEXT,   -- 'user' or 'assistant'
                content         TEXT,
                created_at      REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                telegram_id     TEXT,
                description     TEXT,   -- natural-language description of what to watch for
                ticker          TEXT,
                created_at      REAL,
                active          INTEGER DEFAULT 1
            )
        """)


def get_or_create_user(telegram_id: str, name: str = ""):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE telegram_id=?", (telegram_id,)).fetchone()
        if row:
            return dict(row)
        conn.execute(
            "INSERT INTO users (telegram_id, name, created_at) VALUES (?, ?, ?)",
            (telegram_id, name, time.time()),
        )
        row = conn.execute("SELECT * FROM users WHERE telegram_id=?", (telegram_id,)).fetchone()
        return dict(row)


def update_user(telegram_id: str, **fields):
    if not fields:
        return
    # JSON-encode list fields
    for key in ("watchlist", "interests"):
        if key in fields and not isinstance(fields[key], str):
            fields[key] = json.dumps(fields[key])
    cols = ", ".join(f"{k}=?" for k in fields)
    vals = list(fields.values()) + [telegram_id]
    with get_conn() as conn:
        conn.execute(f"UPDATE users SET {cols} WHERE telegram_id=?", vals)


def add_message(telegram_id: str, role: str, content: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO messages (telegram_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (telegram_id, role, content, time.time()),
        )


def get_recent_messages(telegram_id: str, limit: int = 20):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE telegram_id=? ORDER BY id DESC LIMIT ?",
            (telegram_id, limit),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def all_users_with_briefing():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM users WHERE onboarded=1 AND briefing_time IS NOT NULL"
        ).fetchall()
    return [dict(r) for r in rows]


def get_watchlist(telegram_id: str):
    with get_conn() as conn:
        row = conn.execute("SELECT watchlist FROM users WHERE telegram_id=?", (telegram_id,)).fetchone()
    if not row or not row["watchlist"]:
        return []
    try:
        return json.loads(row["watchlist"])
    except json.JSONDecodeError:
        return []
