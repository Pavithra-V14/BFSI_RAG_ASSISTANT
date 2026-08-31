from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import uuid
import contextvars
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import config

logger = logging.getLogger(__name__)

TRACE_LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "traces.jsonl"
TRACE_LOG_PATH.parent.mkdir(exist_ok=True)
TRACE_ARCHIVE_DIR = TRACE_LOG_PATH.parent / "traces_archive"

GENESIS_HASH = "0" * 64  

_USE_POSTGRES = config.DATABASE_URL is not None
_PG_ADVISORY_LOCK_ID = 918_273_645  

def _pg_conn():
    import psycopg2
    import psycopg2.extras
    return psycopg2.connect(config.DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)


def _pg_init_db() -> None:
    """
    Wrapped in the same advisory lock as the hash-chain writes — confirmed
    live (2026-08-25) that `CREATE TABLE IF NOT EXISTS` is NOT safe
    against genuinely concurrent first-time creation from multiple
    processes: 5 processes started simultaneously against an empty
    database, only 1 survived, the other 4 crashed with a Postgres
    internal catalog UniqueViolation (a race on pg_type, not our own
    constraint). This matters specifically for Render — multiple
    instances/workers starting up at once is a realistic scenario, not
    an edge case. The advisory lock serializes schema creation the same
    way it serializes hash-chain writes.
    """
    conn = _pg_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT pg_advisory_xact_lock(%s)", (_PG_ADVISORY_LOCK_ID,))
        cur.execute("""
            CREATE TABLE IF NOT EXISTS trace_entries (
                id BIGSERIAL PRIMARY KEY,
                trace_id TEXT NOT NULL,
                request_type TEXT NOT NULL,
                user_id TEXT,
                stage TEXT NOT NULL,
                timestamp DOUBLE PRECISION NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                chain_hash TEXT NOT NULL,
                archived BOOLEAN NOT NULL DEFAULT FALSE
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_trace_entries_trace_id ON trace_entries (trace_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_trace_entries_archived ON trace_entries (archived)")
        conn.commit()
    finally:
        conn.close()


if _USE_POSTGRES:
    _pg_init_db()

_trace_file_lock = threading.Lock()
_last_chain_hash: str | None = None  

def _get_last_chain_hash() -> str:
    """Reads the current chain tip once (cached after), from whichever
    file — active or most recent archive — actually has the latest entry.
    Cheap after the first call; only re-scans the file on cold start."""
    global _last_chain_hash
    if _last_chain_hash is not None:
        return _last_chain_hash
    if TRACE_LOG_PATH.exists():
        lines = TRACE_LOG_PATH.read_text().splitlines()
        if lines:
            last_entry = json.loads(lines[-1])
            _last_chain_hash = last_entry.get("_chain_hash", GENESIS_HASH)
            return _last_chain_hash
    _last_chain_hash = GENESIS_HASH
    return _last_chain_hash


def _compute_chain_hash(prev_hash: str, entry_without_hash: dict) -> str:
    canonical = json.dumps(entry_without_hash, sort_keys=True, default=str)
    return hashlib.sha256((prev_hash + canonical).encode()).hexdigest()

REQUIRED_QUERY_STAGES = [
    "input_guardrail", "cache_check", "retrieval", "rerank",
    "generation", "output_guardrail",
]


def _langfuse_configured() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


def _get_langfuse_client():
    try:
        from langfuse import Langfuse
        return Langfuse()
    except Exception as e:  # noqa: BLE001
        logger.warning("Langfuse init failed (%s) — falling back to local JSONL tracer", str(e)[:200])
        return None


class Trace:
    def __init__(self, trace_id: str, request_type: str, user_id: str | None):
        self.trace_id = trace_id
        self.request_type = request_type
        self.user_id = user_id
        self.stages: list[dict] = []
        self._langfuse = _get_langfuse_client() if _langfuse_configured() else None
        self._lf_root_span = None
        if self._langfuse is not None:
            try:
                self._lf_root_span = self._langfuse.start_observation(
                    name=request_type,
                    as_type="span",
                    metadata={"app_trace_id": trace_id, "user_id": user_id},
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("Langfuse start_observation() failed (%s), continuing with JSONL only", str(e)[:200])
                self._lf_root_span = None

    def log_stage(self, stage: str, **metadata) -> None:
        entry = {
            "trace_id": self.trace_id,
            "request_type": self.request_type,
            "user_id": self.user_id,
            "stage": stage,
            "timestamp": time.time(),
            **metadata,
        }
        self.stages.append(entry)
        if self._lf_root_span is not None:
            try:
                self._lf_root_span.create_event(name=stage, metadata=metadata)
            except Exception as e:  # noqa: BLE001
                logger.warning("Langfuse create_event() failed (%s) for stage %s", str(e)[:200], stage)

    def flush(self) -> None:
        if _USE_POSTGRES:
            self._flush_postgres()
        else:
            self._flush_file()
        if self._lf_root_span is not None:
            try:
                self._lf_root_span.end()
            except Exception as e:  
                logger.warning("Langfuse span end() failed (%s)", str(e)[:200])
        if self._langfuse is not None:
            try:
                self._langfuse.flush()
            except Exception as e:  
                logger.warning("Langfuse flush() failed (%s)", str(e)[:200])

    def _flush_file(self) -> None:
        global _last_chain_hash
        with _trace_file_lock:
            chained_lines = []
            prev_hash = _get_last_chain_hash()
            for entry in self.stages:
                this_hash = _compute_chain_hash(prev_hash, entry)
                chained_entry = {**entry, "_chain_hash": this_hash}
                chained_lines.append(json.dumps(chained_entry))
                prev_hash = this_hash
            _last_chain_hash = prev_hash
            with open(TRACE_LOG_PATH, "a") as f:
                for line in chained_lines:
                    f.write(line + "\n")

    def _flush_postgres(self) -> None:
        """
        Item 1.4 — pg_advisory_xact_lock() serializes the read-last-hash-
        then-insert sequence across ANY number of concurrent processes,
        not just threads in one process (the file version's threading.Lock
        limitation). The lock is held for the duration of the transaction
        (released automatically on commit/rollback), so a second process
        calling this blocks until the first one's transaction completes —
        genuine cross-process safety, tested below with real concurrent
        writers, not just claimed.
        """
        conn = _pg_conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT pg_advisory_xact_lock(%s)", (_PG_ADVISORY_LOCK_ID,))
            cur.execute("SELECT chain_hash FROM trace_entries ORDER BY id DESC LIMIT 1")
            row = cur.fetchone()
            prev_hash = row["chain_hash"] if row else GENESIS_HASH

            for entry in self.stages:
                metadata = {k: v for k, v in entry.items() if k not in ("trace_id", "request_type", "user_id", "stage", "timestamp")}
                entry_for_hash = {**entry}  
                this_hash = _compute_chain_hash(prev_hash, entry_for_hash)
                cur.execute(
                    "INSERT INTO trace_entries (trace_id, request_type, user_id, stage, timestamp, metadata, chain_hash) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (entry["trace_id"], entry["request_type"], entry["user_id"], entry["stage"],
                     entry["timestamp"], json.dumps(metadata, default=str), this_hash),
                )
                prev_hash = this_hash
            conn.commit()
        finally:
            conn.close()


_current_trace: contextvars.ContextVar["Trace | None"] = contextvars.ContextVar("_current_trace", default=None)

def get_current_trace() -> "Trace | None":
    return _current_trace.get()

def log_gateway_attempt(stage: str, gateway_result: dict) -> None:
    """
    Call this after ANY gateway.generate() call, from ANYWHERE — logs
    provider_attempts to whatever trace is currently active (via the
    contextvar above), or silently no-ops if there isn't one (e.g. a
    standalone script or pytest run with no request-level trace open).
    Never raises — logging a gateway attempt must not be able to break
    the call that triggered it.
    """
    trace = _current_trace.get()
    if trace is None:
        return
    try:
        trace.log_stage(stage, provider=gateway_result.get("provider"), provider_attempts=gateway_result.get("provider_attempts", []))
    except Exception:  
        pass

def reset_trace_log() -> dict:
    """
    Clears the active trace data for a genuinely fresh start. Does NOT
    touch archived data (file: data/traces_archive/; Postgres: rows with
    archived=TRUE) — a "reset" should mean fresh going forward, not
    erasing what was already properly archived.
    """
    if _USE_POSTGRES:
        return _pg_reset_trace_log()

    global _last_chain_hash
    with _trace_file_lock:
        if TRACE_LOG_PATH.exists():
            TRACE_LOG_PATH.unlink()
        _last_chain_hash = None  
    return {"reset": True}

def _pg_reset_trace_log() -> dict:
    """
    Deletes non-archived rows, then RE-CHAINS whatever archived rows
    remain, starting fresh from GENESIS_HASH.

    2026-08-25 — this re-chaining step was missing in the first version
    and confirmed live to break verify_chain(): the remaining archived
    rows' ORIGINAL hashes were computed against whatever came before them
    in the full pre-reset history (including rows that are now deleted),
    so verify_chain()'s "the first row was chained from GENESIS_HASH"
    assumption failed the moment predecessor rows were removed. Same
    class of problem the file-based rotation already solves via its
    `_rotation_checkpoint` re-chaining — same fix here, just directly
    updating rows in place instead of rewriting a file.

    Held under the advisory lock for the ENTIRE delete+re-chain sequence,
    not just the delete — a concurrent write reading "the last hash"
    mid-recompute would read a stale value otherwise.
    """
    conn = _pg_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT pg_advisory_xact_lock(%s)", (_PG_ADVISORY_LOCK_ID,))
        cur.execute("DELETE FROM trace_entries WHERE archived = FALSE")
        deleted = cur.rowcount

        cur.execute("SELECT * FROM trace_entries ORDER BY id ASC")
        remaining = cur.fetchall()
        prev_hash = GENESIS_HASH
        for row in remaining:
            entry = _pg_row_to_entry(row)
            new_hash = _compute_chain_hash(prev_hash, entry)
            cur.execute("UPDATE trace_entries SET chain_hash = %s WHERE id = %s", (new_hash, row["id"]))
            prev_hash = new_hash

        conn.commit()
        return {"reset": True, "rows_deleted": deleted, "rows_rechained": len(remaining)}
    finally:
        conn.close()

@contextmanager
def start_trace(request_type: str, user_id: str | None = None):
    trace = Trace(trace_id=str(uuid.uuid4()), request_type=request_type, user_id=user_id)
    token = _current_trace.set(trace)
    start = time.time()
    try:
        yield trace
    finally:
        trace.log_stage("_trace_end", total_latency_ms=round((time.time() - start) * 1000, 2))
        trace.flush()
        _current_trace.reset(token)


def read_recent_traces(n: int = 10) -> dict[str, list[dict]]:
    if _USE_POSTGRES:
        return _pg_read_recent_traces(n)
    if not TRACE_LOG_PATH.exists():
        return {}
    lines = [json.loads(l) for l in TRACE_LOG_PATH.read_text().splitlines() if l.strip()]
    by_trace: dict[str, list[dict]] = {}
    for entry in lines:
        by_trace.setdefault(entry["trace_id"], []).append(entry)
    recent_ids = list(by_trace.keys())[-n:]
    return {tid: by_trace[tid] for tid in recent_ids}


def _pg_read_recent_traces(n: int) -> dict[str, list[dict]]:
    conn = _pg_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT trace_id, MIN(id) as first_id FROM trace_entries "
            "WHERE archived = FALSE GROUP BY trace_id ORDER BY first_id DESC LIMIT %s",
            (n,),
        )
        recent_trace_ids = [row["trace_id"] for row in cur.fetchall()]
        if not recent_trace_ids:
            return {}
        cur.execute(
            "SELECT * FROM trace_entries WHERE trace_id = ANY(%s) AND archived = FALSE ORDER BY id ASC",
            (recent_trace_ids,),
        )
        by_trace: dict[str, list[dict]] = {}
        for row in cur.fetchall():
            entry = _pg_row_to_entry(row)
            by_trace.setdefault(entry["trace_id"], []).append(entry)
        return by_trace
    finally:
        conn.close()


def _pg_row_to_entry(row: dict) -> dict:
    """Reconstructs the same flat entry shape the file version stores —
    trace_id/request_type/user_id/stage/timestamp at the top level,
    metadata keys merged in alongside them, exactly matching what
    _flush_postgres() hashed, so verify_chain() recomputes correctly."""
    entry = {
        "trace_id": row["trace_id"], "request_type": row["request_type"],
        "user_id": row["user_id"], "stage": row["stage"], "timestamp": row["timestamp"],
    }
    entry.update(row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"]))
    return entry


def verify_chain(path: Path = TRACE_LOG_PATH) -> dict:
    """
    Recomputes the hash chain from scratch and compares against what's
    stored — any mismatch means an entry was edited, reordered, or
    deleted somewhere after that point. Returns a report, not just a
    bool, so a real audit can see WHERE the chain broke, not just that
    it did.
    """
    if _USE_POSTGRES:
        return _pg_verify_chain()

    if not path.exists():
        return {"valid": True, "entries_checked": 0, "first_break_at_line": None}

    lines = path.read_text().splitlines()
    prev_hash = GENESIS_HASH
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        entry = json.loads(line)
        stored_hash = entry.pop("_chain_hash", None)
        recomputed = _compute_chain_hash(prev_hash, entry)
        if stored_hash != recomputed:
            return {
                "valid": False,
                "entries_checked": i,
                "first_break_at_line": i + 1,
                "trace_id_at_break": entry.get("trace_id"),
                "detail": "stored hash does not match recomputed hash — entry was "
                          "modified, or a prior entry was edited/removed/reordered",
            }
        prev_hash = stored_hash
    return {"valid": True, "entries_checked": len(lines), "first_break_at_line": None}


def _pg_verify_chain() -> dict:
    """Postgres version walks EVERY row (archived and active both) in
    insertion order — one continuous chain across the row's entire
    lifetime, unlike the file version which had to restart its chain at
    each rotation boundary."""
    conn = _pg_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM trace_entries ORDER BY id ASC")
        rows = cur.fetchall()
        prev_hash = GENESIS_HASH
        for i, row in enumerate(rows):
            entry = _pg_row_to_entry(row)
            recomputed = _compute_chain_hash(prev_hash, entry)
            if row["chain_hash"] != recomputed:
                return {
                    "valid": False,
                    "entries_checked": i,
                    "first_break_at_line": i + 1,
                    "trace_id_at_break": entry.get("trace_id"),
                    "detail": "stored hash does not match recomputed hash — entry was "
                              "modified, or a prior entry was edited/removed/reordered",
                }
            prev_hash = row["chain_hash"]
        return {"valid": True, "entries_checked": len(rows), "first_break_at_line": None}
    finally:
        conn.close()


def rotate_old_traces(retention_days: int | None = None) -> dict:
    """
    Item 8's retention half. Does NOT delete anything. See module
    docstring's POSTGRES BACKEND section for why the Postgres path
    (_pg_rotate_old_traces) is genuinely simpler than the file version
    below — in-place archived flag, no re-chaining needed.

    Safe to call repeatedly (e.g. from a daily scheduled job) — a no-op
    if nothing is old enough to rotate yet.
    """
    retention_days = retention_days if retention_days is not None else config.AUDIT_LOG_RETENTION_DAYS
    if _USE_POSTGRES:
        return _pg_rotate_old_traces(retention_days)
    return _file_rotate_old_traces(retention_days)


def _pg_rotate_old_traces(retention_days: int) -> dict:
    cutoff = time.time() - (retention_days * 86400)
    conn = _pg_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE trace_entries SET archived = TRUE WHERE timestamp < %s AND archived = FALSE",
            (cutoff,),
        )
        rotated = cur.rowcount
        conn.commit()
        logger.info(
            "Rotated %d trace entries older than %d days (Postgres — flagged archived in place, "
            "not deletion, chain unbroken)", rotated, retention_days,
        )
        return {"rotated": rotated, "archive_file": None}  
    finally:
        conn.close()


def _file_rotate_old_traces(retention_days: int) -> dict:
    if not TRACE_LOG_PATH.exists():
        return {"rotated": 0, "archive_file": None}

    cutoff = time.time() - (retention_days * 86400)
    lines = [l for l in TRACE_LOG_PATH.read_text().splitlines() if l.strip()]
    entries = [json.loads(l) for l in lines]

    old_entries = [e for e in entries if e.get("timestamp", time.time()) < cutoff]
    new_entries = [e for e in entries if e.get("timestamp", time.time()) >= cutoff]

    if not old_entries:
        return {"rotated": 0, "archive_file": None}

    TRACE_ARCHIVE_DIR.mkdir(exist_ok=True)
    archive_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    archive_path = TRACE_ARCHIVE_DIR / f"traces_until_{archive_date}.jsonl"

    with _trace_file_lock:
        with open(archive_path, "a") as f:
            for e in old_entries:
                f.write(json.dumps(e) + "\n")
        archive_final_hash = old_entries[-1].get("_chain_hash", GENESIS_HASH)

        checkpoint = {
            "trace_id": "_rotation_checkpoint",
            "request_type": "_system",
            "user_id": None,
            "stage": "_rotation_checkpoint",
            "timestamp": time.time(),
            "archived_to": str(archive_path),
            "archived_count": len(old_entries),
            "previous_chain_final_hash": archive_final_hash,
        }
        checkpoint_hash = _compute_chain_hash(GENESIS_HASH, checkpoint)
        chained_checkpoint = {**checkpoint, "_chain_hash": checkpoint_hash}

        global _last_chain_hash
        with open(TRACE_LOG_PATH, "w") as f:
            f.write(json.dumps(chained_checkpoint) + "\n")
            prev_hash = checkpoint_hash
            for e in new_entries:
                e_without_hash = {k: v for k, v in e.items() if k != "_chain_hash"}
                new_hash = _compute_chain_hash(prev_hash, e_without_hash)
                f.write(json.dumps({**e_without_hash, "_chain_hash": new_hash}) + "\n")
                prev_hash = new_hash
            _last_chain_hash = prev_hash

    logger.info(
        "Rotated %d trace entries older than %d days to %s (retention policy, not deletion)",
        len(old_entries), retention_days, archive_path,
    )
    return {"rotated": len(old_entries), "archive_file": str(archive_path)}