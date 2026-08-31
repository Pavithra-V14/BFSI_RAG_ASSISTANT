from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import config

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "app.db"
DB_PATH.parent.mkdir(exist_ok=True)

VALID_ROLES = {"claims_adjuster", "compliance_officer", "admin"}
SELF_SERVE_ROLES = {"claims_adjuster"}  

_USE_POSTGRES = config.DATABASE_URL is not None
_PG_ADVISORY_LOCK_ID = 918_273_646  
_PG_FIRST_ADMIN_LOCK_ID = 918_273_649 

@dataclass
class User:
    id: int
    username: str
    email: str
    role: str


def _translate_placeholders(sql: str) -> str:
    """%s (Postgres-native, what every query below is written in) -> ?
    (SQLite) for the local-dev backend. Safe: no query in this module
    contains a literal '%s' substring outside of parameter placeholders."""
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
    """Runs a %s-placeholder query against whichever backend is active,
    translating placeholders for SQLite automatically. Returns the cursor
    (call .fetchone()/.fetchall() as needed)."""
    cur = conn.cursor()
    cur.execute(sql if _USE_POSTGRES else _translate_placeholders(sql), params)
    return cur


def init_db() -> None:
    with _conn() as conn:
        if _USE_POSTGRES:
            _execute(conn, "SELECT pg_advisory_xact_lock(%s)", (_PG_ADVISORY_LOCK_ID,))
            _execute(conn, """
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    failed_login_count INTEGER NOT NULL DEFAULT 0,
                    locked_until DOUBLE PRECISION
                )
            """)
        else:
            _execute(conn, """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    failed_login_count INTEGER NOT NULL DEFAULT 0,
                    locked_until REAL
                )
            """)

        id_type = "SERIAL PRIMARY KEY" if _USE_POSTGRES else "INTEGER PRIMARY KEY AUTOINCREMENT"
        float_type = "DOUBLE PRECISION" if _USE_POSTGRES else "REAL"

        _execute(conn, f"""
            CREATE TABLE IF NOT EXISTS revoked_tokens (
                jti TEXT PRIMARY KEY,
                revoked_at {float_type} NOT NULL,
                expires_at {float_type} NOT NULL
            )
        """)
        _execute(conn, f"""
            CREATE TABLE IF NOT EXISTS password_reset_tokens (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at {float_type} NOT NULL,
                expires_at {float_type} NOT NULL,
                used INTEGER NOT NULL DEFAULT 0
            )
        """)

        if _USE_POSTGRES:
            cur = _execute(conn, "SELECT column_name FROM information_schema.columns WHERE table_name = 'users'")
            existing_cols = {row["column_name"] for row in cur.fetchall()}
        else:
            cur = _execute(conn, "PRAGMA table_info(users)")
            existing_cols = {row["name"] for row in cur.fetchall()}
        if "failed_login_count" not in existing_cols:
            _execute(conn, "ALTER TABLE users ADD COLUMN failed_login_count INTEGER NOT NULL DEFAULT 0")
        if "locked_until" not in existing_cols:
            _execute(conn, f"ALTER TABLE users ADD COLUMN locked_until {float_type}")


def create_user(username: str, email: str, password_hash: str, role: str) -> User:
    if role not in VALID_ROLES:
        raise ValueError(f"invalid role '{role}', must be one of {sorted(VALID_ROLES)}")
    with _conn() as conn:
        cur = _execute(
            conn,
            "INSERT INTO users (username, email, password_hash, role) VALUES (%s, %s, %s, %s) RETURNING id",
            (username, email, password_hash, role),
        )
        new_id = cur.fetchone()["id"]
        return User(id=new_id, username=username, email=email, role=role)


def get_user_by_username(username: str):
    with _conn() as conn:
        cur = _execute(conn, "SELECT * FROM users WHERE username = %s", (username,))
        return cur.fetchone()


def get_user_by_id(user_id: int):
    with _conn() as conn:
        cur = _execute(conn, "SELECT * FROM users WHERE id = %s", (user_id,))
        return cur.fetchone()


def username_or_email_exists(username: str, email: str) -> bool:
    with _conn() as conn:
        cur = _execute(conn, "SELECT 1 FROM users WHERE username = %s OR email = %s", (username, email))
        return cur.fetchone() is not None


def update_role(user_id: int, new_role: str) -> bool:
    """Admin-only operation (enforced at the route layer via require_role,
    not here) — this function trusts the caller has already authorized
    the change. Returns False if the user doesn't exist or the role is
    invalid, True on success."""
    if new_role not in VALID_ROLES:
        raise ValueError(f"invalid role '{new_role}', must be one of {sorted(VALID_ROLES)}")
    with _conn() as conn:
        cur = _execute(conn, "UPDATE users SET role = %s WHERE id = %s", (new_role, user_id))
        return cur.rowcount > 0


def _normalize_row(row: dict) -> dict:
    """
    Postgres's psycopg2 driver returns
    TIMESTAMPTZ columns as real Python datetime objects; SQLite returns
    the same logical column as a plain string (it's declared TEXT there).
    Every Pydantic response model in app.py expects created_at as a str
    (worked fine against SQLite for months, broke immediately against
    Postgres with a real ValidationError). Normalized HERE, at the store
    layer, not in every caller — consistent with this whole migration's
    design: callers should never need to know or care which backend is
    active.
    """
    d = dict(row)
    if "created_at" in d and hasattr(d["created_at"], "isoformat"):
        d["created_at"] = d["created_at"].isoformat()
    return d


def list_users():
    with _conn() as conn:
        cur = _execute(conn, "SELECT id, username, email, role, created_at FROM users ORDER BY id")
        return [_normalize_row(row) for row in cur.fetchall()]


def admin_exists() -> bool:
    with _conn() as conn:
        cur = _execute(conn, "SELECT 1 FROM users WHERE role = 'admin' LIMIT 1")
        return cur.fetchone() is not None


def promote_if_first_user(user_id: int) -> bool:
    """
    Race-safe "first user becomes admin" check — the alternative bootstrap
    path to scripts/create_admin.py (see config.AUTO_PROMOTE_FIRST_USER_
    TO_ADMIN). Returns True if promotion happened.

    Wrapped in the same pg_advisory_xact_lock() pattern proven necessary
    today (tracer.py's migration testing found CREATE TABLE IF NOT EXISTS
    genuinely racy under real concurrent processes) — without this lock,
    two users registering at the exact same moment on a fresh deployment
    could BOTH see "no admin exists yet" and both get promoted, which
    defeats the entire point of "the first one" being special. On
    SQLite, the file-level locking sqlite3 already does for writes is
    sufficient (no separate advisory-lock concept exists there, and this
    codebase is single-process for the SQLite path anyway).
    """
    if not config.AUTO_PROMOTE_FIRST_USER_TO_ADMIN:
        return False
    with _conn() as conn:
        if _USE_POSTGRES:
            _execute(conn, "SELECT pg_advisory_xact_lock(%s)", (_PG_FIRST_ADMIN_LOCK_ID,))
        cur = _execute(conn, "SELECT 1 FROM users WHERE role = 'admin' LIMIT 1")
        if cur.fetchone() is not None:
            return False  # an admin already exists — not the first user, no promotion
        _execute(conn, "UPDATE users SET role = 'admin' WHERE id = %s", (user_id,))
        return True

MAX_FAILED_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION_SECONDS = 15 * 60  

def is_locked_out(username: str) -> bool:
    """Checked BEFORE password verification in /auth/login — a locked
    account must not even attempt the password check, or the lockout
    provides no real brute-force protection."""
    import time
    row = get_user_by_username(username)
    if row is None:
        return False
    locked_until = row["locked_until"]
    if locked_until is None:
        return False
    if time.time() > locked_until:
        _clear_lockout(row["id"])
        return False
    return True


def record_failed_login(username: str) -> None:
    import time
    row = get_user_by_username(username)
    if row is None:
        return 
    new_count = row["failed_login_count"] + 1
    with _conn() as conn:
        if new_count >= MAX_FAILED_LOGIN_ATTEMPTS:
            locked_until = time.time() + LOCKOUT_DURATION_SECONDS
            _execute(
                conn, "UPDATE users SET failed_login_count = %s, locked_until = %s WHERE id = %s",
                (new_count, locked_until, row["id"]),
            )
        else:
            _execute(conn, "UPDATE users SET failed_login_count = %s WHERE id = %s", (new_count, row["id"]))


def record_successful_login(user_id: int) -> None:
    _clear_lockout(user_id)


def _clear_lockout(user_id: int) -> None:
    with _conn() as conn:
        _execute(conn, "UPDATE users SET failed_login_count = 0, locked_until = NULL WHERE id = %s", (user_id,))

def revoke_token(jti: str, expires_at: float) -> None:
    """expires_at: the token's own expiry — once past it, the revocation
    record is dead weight (an expired token is already unusable), so
    purge_expired_revocations() can safely clean these up on that basis."""
    import time
    with _conn() as conn:
        if _USE_POSTGRES:
            _execute(
                conn,
                "INSERT INTO revoked_tokens (jti, revoked_at, expires_at) VALUES (%s, %s, %s) "
                "ON CONFLICT (jti) DO UPDATE SET revoked_at = EXCLUDED.revoked_at, expires_at = EXCLUDED.expires_at",
                (jti, time.time(), expires_at),
            )
        else:
            _execute(
                conn, "INSERT OR REPLACE INTO revoked_tokens (jti, revoked_at, expires_at) VALUES (%s, %s, %s)",
                (jti, time.time(), expires_at),
            )

def is_token_revoked(jti: str) -> bool:
    with _conn() as conn:
        cur = _execute(conn, "SELECT 1 FROM revoked_tokens WHERE jti = %s", (jti,))
        return cur.fetchone() is not None


def purge_expired_revocations() -> int:
    import time
    with _conn() as conn:
        cur = _execute(conn, "DELETE FROM revoked_tokens WHERE expires_at < %s", (time.time(),))
        return cur.rowcount

PASSWORD_RESET_TOKEN_TTL_SECONDS = 30 * 60 

def create_password_reset_token(user_id: int) -> str:
    import secrets
    import time
    token = secrets.token_urlsafe(32)
    now = time.time()
    with _conn() as conn:
        _execute(
            conn,
            "INSERT INTO password_reset_tokens (token, user_id, created_at, expires_at, used) "
            "VALUES (%s, %s, %s, %s, 0)",
            (token, user_id, now, now + PASSWORD_RESET_TOKEN_TTL_SECONDS),
        )
    return token


def consume_password_reset_token(token: str) -> int | None:
    """Validates and marks the token used in one step — returns the
    user_id if valid, None if invalid/expired/already-used. Single-use by
    design: a reset token must not be replayable."""
    import time
    with _conn() as conn:
        cur = _execute(
            conn, "SELECT user_id, expires_at, used FROM password_reset_tokens WHERE token = %s", (token,)
        )
        row = cur.fetchone()
        if row is None or row["used"] or row["expires_at"] < time.time():
            return None
        _execute(conn, "UPDATE password_reset_tokens SET used = 1 WHERE token = %s", (token,))
        return row["user_id"]


def update_password(user_id: int, new_password_hash: str) -> None:
    with _conn() as conn:
        _execute(conn, "UPDATE users SET password_hash = %s WHERE id = %s", (new_password_hash, user_id))


init_db()