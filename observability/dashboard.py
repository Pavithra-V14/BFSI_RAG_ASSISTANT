"""
Dashboard metrics — the "glance and know if something's wrong" numbers
promised in the architecture discussion (CPSO, latency p95, faithfulness
rate, fail-closed-trigger rate), computed from the trace log rather than
guessed or left as an aspirational design doc.

Reads the same trace source `observability.trace_check` reads (local
JSONL, always dual-written even when Langfuse is also configured — see
tracer.py) so this works identically whether or not Langfuse is set up.
This is the one place that turns "we log everything" into "here's what
the logs actually say," which is the gap between a good architecture doc
and a dashboard someone can actually look at.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

import config
from observability.tracer import read_recent_traces

FEEDBACK_LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "feedback.jsonl"
EVAL_HISTORY_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_history.jsonl"

_USE_POSTGRES = config.DATABASE_URL is not None
_PG_ADVISORY_LOCK_ID = 918_273_648  # distinct from auth (…646), sessions (…647),
# tracer (…645) — each module serializes its own schema creation independently.


def _pg_conn():
    import psycopg2
    import psycopg2.extras
    return psycopg2.connect(config.DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)


def _pg_init_db() -> None:
    """
    2026-08-25 — item 1.3's follow-up: feedback.jsonl and eval_history.jsonl
    were ALSO local files that would be wiped on Render's free tier, same
    problem traces.jsonl had — found while checking app.py for other file
    references during the tracer.py migration, not originally scoped, but
    leaving it out would have been exactly the kind of silent gap this
    whole migration exists to close. Simpler than trace_entries: no
    hash-chaining needed, these were never a tamper-evident audit trail,
    just append-only records for dashboard aggregation. Advisory-lock
    wrapped for the same concurrent-first-creation race confirmed real in
    tracer.py's migration testing.
    """
    conn = _pg_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT pg_advisory_xact_lock(%s)", (_PG_ADVISORY_LOCK_ID,))
        cur.execute("""
            CREATE TABLE IF NOT EXISTS feedback_entries (
                id BIGSERIAL PRIMARY KEY,
                trace_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                user_id INTEGER,
                rating TEXT NOT NULL,
                comment TEXT,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS eval_history (
                id BIGSERIAL PRIMARY KEY,
                summary JSONB NOT NULL,
                recorded_at DOUBLE PRECISION NOT NULL
            )
        """)
        conn.commit()
    finally:
        conn.close()


if _USE_POSTGRES:
    _pg_init_db()


@dataclass
class DashboardMetrics:
    request_count: int
    cache_hit_rate: float | None
    cache_hit_rate_by_role: dict[str, float]
    refusal_rate: float | None          # input guardrail refuse/block
    fail_closed_rate: float | None      # empty retrieval / ungrounded / context-guardrail failure
    grounded_rate: float | None         # fraction of non-short-circuited requests that shipped grounded
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    latency_p99_ms: float | None
    stage_latency_p50_ms: dict[str, float]   # median time spent BEFORE each stage started, per request
    total_estimated_cost: float
    avg_cost_per_request: float | None
    cpso: float | None                  # cost per successful (grounded) outcome
    pii_masked_count: int
    context_guardrail_drop_count: int
    stale_citation_count: int           # should always be 0 — see app.py's belt-and-suspenders check
    provider_reliability: dict[str, dict[str, int]]  # provider -> {success, error, rate_limited, circuit_open}
    feedback_up_count: int
    feedback_down_count: int
    feedback_ratio: float | None        # up / (up + down)


def _stage_map(stages: list[dict]) -> dict[str, dict]:
    """First occurrence of each stage name -> its logged fields, per trace."""
    out: dict[str, dict] = {}
    for s in stages:
        if s["stage"] not in out:
            out[s["stage"]] = s
    return out


def record_feedback(trace_id: str, session_id: str, user_id: int, rating: str, comment: str | None) -> None:
    """
    2026-08-25 — new. Previously app.py's /feedback endpoint wrote
    directly to a raw file path, bypassing this module entirely — a real
    architectural gap (the ONE place that should own how feedback is
    persisted didn't), found while auditing for other Render-blocking
    file references during the tracer.py migration. Fixed properly here
    rather than left as a parallel, un-migrated write path.
    """
    if _USE_POSTGRES:
        conn = _pg_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO feedback_entries (trace_id, session_id, user_id, rating, comment) "
                "VALUES (%s, %s, %s, %s, %s)",
                (trace_id, session_id, user_id, rating, comment),
            )
            conn.commit()
        finally:
            conn.close()
        return

    import time
    FEEDBACK_LOG_PATH.parent.mkdir(exist_ok=True)
    entry = {
        "trace_id": trace_id, "session_id": session_id, "user_id": user_id,
        "rating": rating, "comment": comment, "timestamp": time.time(),
    }
    with open(FEEDBACK_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _read_feedback_counts() -> tuple[int, int]:
    if _USE_POSTGRES:
        conn = _pg_conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT rating, COUNT(*) as c FROM feedback_entries GROUP BY rating")
            counts = {row["rating"]: row["c"] for row in cur.fetchall()}
            return counts.get("up", 0), counts.get("down", 0)
        finally:
            conn.close()

    if not FEEDBACK_LOG_PATH.exists():
        return 0, 0
    up = down = 0
    for line in FEEDBACK_LOG_PATH.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("rating") == "up":
            up += 1
        elif entry.get("rating") == "down":
            down += 1
    return up, down


def compute_dashboard_metrics(n: int = 200, request_type_filter: str | None = None) -> DashboardMetrics:
    """
    request_type_filter: None = everything (live + eval blended, the
    original behavior). "query" = live traffic only. "eval"/"eval_refusal"
    = evaluation runs only. Added 2026-08-24 — eval traces existed in the
    log all along (item 9's tracing work), but nothing ever separated them
    from live traffic in the aggregate metrics, so a heavy eval run would
    silently skew what looked like "live" cache-hit-rate, latency, cost,
    etc. Two genuinely different things being reported as one number.
    """
    all_traces = read_recent_traces(n * 3 if request_type_filter else n)  # over-fetch when
    # filtering, since a chunk of the last N*3 raw entries may belong to
    # the OTHER request_type and get filtered out — without over-fetching,
    # a heavy eval run right before checking the live dashboard could
    # starve it down to very few or zero live traces to report on.
    if request_type_filter:
        traces = {
            tid: stages for tid, stages in all_traces.items()
            if stages and stages[0].get("request_type", "").startswith(request_type_filter)
        }
        traces = dict(list(traces.items())[-n:])
    else:
        traces = all_traces
    n_traces = len(traces)
    up, down = _read_feedback_counts()
    feedback_ratio = round(up / (up + down), 3) if (up + down) else None

    if n_traces == 0:
        return DashboardMetrics(
            request_count=0, cache_hit_rate=None, cache_hit_rate_by_role={}, refusal_rate=None,
            fail_closed_rate=None, grounded_rate=None, latency_p50_ms=None, latency_p95_ms=None,
            latency_p99_ms=None, stage_latency_p50_ms={}, total_estimated_cost=0.0,
            avg_cost_per_request=None, cpso=None, pii_masked_count=0, context_guardrail_drop_count=0,
            stale_citation_count=0, provider_reliability={},
            feedback_up_count=up, feedback_down_count=down, feedback_ratio=feedback_ratio,
        )

    cache_hits = 0
    cache_by_role: dict[str, list[int]] = {}  # role -> [hit, miss, hit, ...] as 1/0
    refusals = 0
    fail_closed = 0
    grounded = 0
    scoreable = 0  # requests that reached output_guardrail at all (excludes hard refusals)
    latencies = []
    total_cost = 0.0
    pii_masked = 0
    ctx_dropped = 0
    stale_citations = 0
    provider_reliability: dict[str, dict[str, int]] = {}
    stage_offsets: dict[str, list[float]] = {}

    for trace_id, stages in traces.items():
        by_stage = _stage_map(stages)
        start_ts = stages[0]["timestamp"] if stages else None

        cc = by_stage.get("cache_check")
        if cc:
            hit = 1 if cc.get("hit") else 0
            cache_hits += hit
            role = cc.get("role", "unknown")
            cache_by_role.setdefault(role, []).append(hit)

        ig = by_stage.get("input_guardrail")
        if ig and ig.get("action") in ("refuse", "block"):
            refusals += 1

        og = by_stage.get("output_guardrail")
        if og:
            scoreable += 1
            if og.get("grounded"):
                grounded += 1
            reason = og.get("reason", "")
            if og.get("grounded") is False and reason in (
                "empty_retrieval", "insufficient_grounded_context_after_filtering",
            ):
                fail_closed += 1
            stale_citations += og.get("stale_citation_count", 0)

        ctx = by_stage.get("context_guardrail")
        if ctx:
            ctx_dropped += ctx.get("dropped_count", 0)
            pii_masked += ctx.get("pii_masked_count", 0)

        gen = by_stage.get("generation")
        if gen and gen.get("estimated_cost"):
            total_cost += gen["estimated_cost"]

        # 2026-08-24 fix: provider_reliability used to only scan the main
        # "generation" stage, silently missing three OTHER real gateway
        # call sites (context rewrite, domain-relevance classification,
        # the eval faithfulness judge) — confirmed live: the terminal
        # showed heavy Cerebras/Gemini/OpenRouter activity the dashboard
        # never reflected, because those three call sites had nowhere to
        # log to. Scanning all four stage names closes that gap.
        for gateway_stage in ("generation", "context_rewrite_generation", "domain_relevance_classification", "faithfulness_judge"):
            stage_entry = by_stage.get(gateway_stage)
            if not stage_entry:
                continue
            for attempt in stage_entry.get("provider_attempts", []):
                p = attempt.get("provider", "unknown")
                outcome = attempt.get("outcome", "unknown")
                bucket = provider_reliability.setdefault(
                    p, {"success": 0, "error": 0, "rate_limited": 0, "circuit_open": 0}
                )
                bucket[outcome] = bucket.get(outcome, 0) + 1

        end_entry = by_stage.get("_trace_end")
        if end_entry and end_entry.get("total_latency_ms") is not None:
            latencies.append(end_entry["total_latency_ms"])

        if start_ts is not None:
            for s in stages:
                stage_offsets.setdefault(s["stage"], []).append((s["timestamp"] - start_ts) * 1000)

    def pctl(data: list[float], p: float) -> float | None:
        if not data:
            return None
        data = sorted(data)
        idx = min(int(len(data) * p), len(data) - 1)
        return round(data[idx], 2)

    stage_p50 = {
        stage: round(statistics.median(offsets), 2)
        for stage, offsets in stage_offsets.items()
        if stage != "_trace_end"
    }

    grounded_rate = round(grounded / scoreable, 3) if scoreable else None
    avg_cost = round(total_cost / n_traces, 5) if n_traces else None
    cpso = round(total_cost / grounded, 5) if grounded else None
    cache_by_role_rate = {
        role: round(sum(hits) / len(hits), 3) for role, hits in cache_by_role.items()
    }

    return DashboardMetrics(
        request_count=n_traces,
        cache_hit_rate=round(cache_hits / n_traces, 3),
        cache_hit_rate_by_role=cache_by_role_rate,
        refusal_rate=round(refusals / n_traces, 3),
        fail_closed_rate=round(fail_closed / n_traces, 3),
        grounded_rate=grounded_rate,
        latency_p50_ms=pctl(latencies, 0.50),
        latency_p95_ms=pctl(latencies, 0.95),
        latency_p99_ms=pctl(latencies, 0.99),
        stage_latency_p50_ms=stage_p50,
        total_estimated_cost=round(total_cost, 5),
        avg_cost_per_request=avg_cost,
        cpso=cpso,
        pii_masked_count=pii_masked,
        context_guardrail_drop_count=ctx_dropped,
        stale_citation_count=stale_citations,
        provider_reliability=provider_reliability,
        feedback_up_count=up,
        feedback_down_count=down,
        feedback_ratio=feedback_ratio,
    )


def as_dict(n: int = 200, request_type_filter: str | None = None) -> dict:
    return asdict(compute_dashboard_metrics(n, request_type_filter))


def record_eval_run(summary: dict) -> None:
    """Appends one eval run's summary to a history log, timestamped —
    this is what makes drift ("is retrieval quality slipping over time?")
    checkable instead of only ever seeing the single most recent number."""
    import time
    if _USE_POSTGRES:
        conn = _pg_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO eval_history (summary, recorded_at) VALUES (%s, %s)",
                (json.dumps(summary), time.time()),
            )
            conn.commit()
        finally:
            conn.close()
        return

    EVAL_HISTORY_PATH.parent.mkdir(exist_ok=True)
    entry = {**summary, "recorded_at": time.time()}
    with open(EVAL_HISTORY_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def read_eval_history(last_n: int = 20) -> list[dict]:
    if _USE_POSTGRES:
        conn = _pg_conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT summary, recorded_at FROM eval_history ORDER BY id DESC LIMIT %s", (last_n,))
            rows = cur.fetchall()
            results = []
            for row in reversed(rows):  # DESC-fetched for LIMIT, reversed back to chronological order
                s = row["summary"] if isinstance(row["summary"], dict) else json.loads(row["summary"])
                results.append({**s, "recorded_at": row["recorded_at"]})
            return results
        finally:
            conn.close()

    if not EVAL_HISTORY_PATH.exists():
        return []
    lines = [json.loads(l) for l in EVAL_HISTORY_PATH.read_text().splitlines() if l.strip()]
    return lines[-last_n:]


def reset_feedback_and_eval_history() -> dict:
    """
    2026-08-25 — proper reset function, replacing app.py's previous
    direct file-unlink calls, which silently did nothing on the Postgres
    backend (FEEDBACK_LOG_PATH/EVAL_HISTORY_PATH never get written to in
    that mode, so .exists() was always False — reset appeared to succeed
    but the Postgres tables were never actually cleared)."""
    if _USE_POSTGRES:
        conn = _pg_conn()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM feedback_entries")
            cur.execute("DELETE FROM eval_history")
            conn.commit()
        finally:
            conn.close()
        return {"cleared": ["feedback_entries", "eval_history"]}

    cleared = []
    for path in (FEEDBACK_LOG_PATH, EVAL_HISTORY_PATH):
        if path.exists():
            path.unlink()
            cleared.append(path.name)
    return {"cleared": cleared}