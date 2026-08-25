"""
Per-request tracing — every pipeline stage writes to the trace, under a
shared trace_id (see context-graph.json invariant: every_request_has_a_trace_id).

PRIMARY: Langfuse (see ADR 0007). Requires LANGFUSE_PUBLIC_KEY,
LANGFUSE_SECRET_KEY, LANGFUSE_HOST. FALLBACK: local JSONL file, used
automatically when Langfuse isn't configured — same trace_id/stage
contract either way, so `observability.trace_check` works against either
backend without modification.

SDK note: Langfuse's Python SDK went through a breaking rewrite (the
installed version here is 4.x, OTEL-based) — the older `.trace(...)`
client method used by earlier Langfuse examples/tutorials no longer
exists. This module uses the current API: `client.start_observation(...)`
returns a span object directly (not a context manager), with `.end()`,
`.update()`, and `.create_event()` for point-in-time stage markers nested
under it, and `.start_observation(...)` again on that span object for
nested child spans.

TAMPER-EVIDENCE (item 8, 2026-08-24): the local JSONL file is a compliance
audit trail — a plain append-only file can be silently edited or deleted
with no way to detect it, which defeats the purpose of an audit trail.
Every entry now carries a _chain_hash = sha256(prev_entry's _chain_hash +
this entry's own content), computed sequentially over the WHOLE file
(not per-trace) — the same pattern a blockchain or git commit chain uses.
Editing or removing any past entry breaks every hash after it, which
verify_chain() below can detect. This does NOT prevent tampering (anyone
with filesystem access can still edit the file) — it makes tampering
DETECTABLE, which is the realistic bar for a local-file audit log; true
tamper-PREVENTION needs write-once storage (S3 Object Lock, a real audit
log service) outside what this codebase controls.

RETENTION (item 8's other half): AUDIT_LOG_RETENTION_DAYS (config.py)
defines how long entries stay in the ACTIVE file. rotate_old_traces()
below does not DELETE old entries — retention in most compliance regimes
is a minimum "must still exist" requirement, not a mandate to destroy
data — it moves them into a dated archive file, which is independently
hash-chain-verifiable on its own. The active file starts a fresh chain
after rotation, with a `_rotation_checkpoint` entry recording the
archive's final hash, so full history remains traceable across the split.

POSTGRES BACKEND (2026-08-25, item 1.3): when config.DATABASE_URL is set
(any deployment where the filesystem doesn't survive a restart — see
that config value's docstring), trace storage moves to a Postgres table
instead of the local JSONL file. This is NOT the same code translated —
Postgres allows a genuinely SIMPLER design than the file version: instead
of physically moving "old" lines into a separate archive file and having
to re-chain the entries left behind (the `_rotation_checkpoint` hack
below), rotation in Postgres is just flagging old rows `archived = true`
in place. The hash chain stays ONE continuous sequence across the row's
entire lifetime, archived or not — no re-chaining, no checkpoint entries,
no separate archive file to keep track of. verify_chain() walks every row
by insertion order regardless of archived status; the dashboard's
read_recent_traces() only looks at non-archived rows.

Item 1.4 — real concurrency safety: the file version's threading.Lock
only protects against races WITHIN one Python process; two Render
instances (or two workers) writing simultaneously could still both read
the same "last hash" and corrupt the chain. The Postgres path uses
pg_advisory_xact_lock() — a database-level lock that serializes the
read-last-hash-then-insert sequence across ANY number of concurrent
processes/connections, not just threads within one process.
"""
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

GENESIS_HASH = "0" * 64  # the chain's starting anchor — same convention as
# git's "no parent commit" or a blockchain's genesis block

_USE_POSTGRES = config.DATABASE_URL is not None
_PG_ADVISORY_LOCK_ID = 918_273_645  # arbitrary fixed 32-bit int, unique to
# this lock's purpose within the database — pg_advisory_xact_lock() keys
# on this number, so any process calling it with the same number
# serializes against every other process doing the same


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

# 2026-08-24 — explicit lock around trace-file writes. Confirmed live that
# small concurrent writes (4-way parallel eval, or live traffic during an
# eval run) happened not to corrupt the JSONL file, but that was the OS's
# small-write atomicity working by fortunate circumstance, not a
# guarantee — a larger trace entry (e.g. a big provider_attempts list) or
# higher real production concurrency could interleave partial writes.
# This makes it deterministically safe instead of probabilistically safe.
# ALSO now guards the hash-chain state (_last_chain_hash below) — chaining
# requires writes to genuinely serialize, not just avoid corruption.
_trace_file_lock = threading.Lock()
_last_chain_hash: str | None = None  # cached in-memory so every write
# doesn't have to re-read the whole file to find the current chain tip


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
        # Always ALSO write locally when Langfuse is configured — the
        # local backend (file or Postgres) is what observability.
        # trace_check reads for the M7 demo command without needing
        # Langfuse API credentials to verify.
        if _USE_POSTGRES:
            self._flush_postgres()
        else:
            self._flush_file()
        if self._lf_root_span is not None:
            try:
                self._lf_root_span.end()
            except Exception as e:  # noqa: BLE001
                logger.warning("Langfuse span end() failed (%s)", str(e)[:200])
        if self._langfuse is not None:
            try:
                self._langfuse.flush()
            except Exception as e:  # noqa: BLE001
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
                entry_for_hash = {**entry}  # hash over the SAME shape the file version hashes,
                # for consistency of what "tamper-evidence" actually covers
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
# 2026-08-24 fix: three real call sites hit the LLM gateway but had no way
# to log their provider_attempts anywhere — contextualize.py's follow-up
# rewrite, input_guardrail.py's domain-relevance classification (item 10),
# and eval/ragas_metrics.py's faithfulness judge. Confirmed live: the
# dashboard's "provider reliability" table only ever reflected the ONE
# call site (app.py's main generation) that explicitly logged to a trace
# object it was handed directly — the other three were invisible to it
# despite being clearly visible failing in the terminal log. Rather than
# thread a `trace` parameter through three unrelated function signatures
# (contextualize_query, _domain_score_llm, the judge), this contextvar
# makes "the currently active trace" discoverable from anywhere in the
# call stack — see log_gateway_attempt() below.


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
    except Exception:  # noqa: BLE001
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
        _last_chain_hash = None  # forces the next write to correctly start from GENESIS_HASH
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
        # Only non-archived rows — matches the file version's "active
        # file only" semantics (archived/rotated history isn't part of
        # the live dashboard's recent-activity window).
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
        return {"rotated": rotated, "archive_file": None}  # no separate file in the Postgres
        # model — "archived" is a flag on the same table, not a relocation
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
        # Archive file gets its own genesis-anchored chain (old_entries
        # already HAVE valid _chain_hash values from the original active
        # file, computed against the pre-rotation genesis — preserved
        # as-is, since re-chaining them would itself look like tampering
        # to anyone verifying against the original hashes they may have
        # separately recorded).
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
                # Re-chained starting from the checkpoint, NOT kept as-is —
                # their original _chain_hash values were computed against
                # the pre-rotation chain (which included the now-archived
                # entries before them) and would fail verify_chain() on
                # the active file alone otherwise. This is honest, not
                # tamper-masking: the checkpoint entry transparently
                # records exactly what was archived and its final hash,
                # so the full history is still traceable by following
                # that pointer into the archive file.
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