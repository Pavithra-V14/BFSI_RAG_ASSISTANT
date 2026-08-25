"""
Central configuration — every tuned numeric value in the pipeline lives
here, overridable via environment variable, defaulting to the values
already validated in wiki/concepts/pipeline-parameters.md.

Why centralize: these values were previously scattered as local module
constants (RELEVANCE_FLOOR in retrieval/rerank.py, GROUNDEDNESS_THRESHOLD
in guardrails/output_guardrail.py, SIMILARITY_THRESHOLD in
retrieval/cache.py, etc.) — correct individually, but meant tuning any one
of them required a code change and redeploy, and there was no single place
to see every knob at once. Production systems need these adjustable
without a code change (e.g. lowering the groundedness threshold temporarily
during an incident, or raising rate limits for a burst of legitimate
traffic) — that's what env-var overrides are for.

Every value below has a sensible default; nothing here is REQUIRED to be
set. Setting an env var overrides the default; unset means "use the
value already validated against the golden set."
"""
from __future__ import annotations

import os


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    return default if val is None else val.lower() in ("1", "true", "yes")


def _list(name: str, default: list[str]) -> list[str]:
    val = os.environ.get(name)
    return [v.strip() for v in val.split(",")] if val else default


# ── Retrieval ──
RETRIEVAL_TOP_K = _int("RETRIEVAL_TOP_K", 25)
RERANK_TOP_K = _int("RERANK_TOP_K", 5)
RRF_K = _int("RRF_K", 60)

# ── Guardrails ──
RELEVANCE_FLOOR = _float("RELEVANCE_FLOOR", 0.30)
GROUNDEDNESS_THRESHOLD = _float("GROUNDEDNESS_THRESHOLD", 0.35)
CONTEXT_MIN_RELEVANCE_SCORE = _float("CONTEXT_MIN_RELEVANCE_SCORE", 0.02)
OUT_OF_DOMAIN_CONFIDENCE_FLOOR = _float("OUT_OF_DOMAIN_CONFIDENCE_FLOOR", 0.15)
DOMAIN_RELEVANCE_FLOOR = _float("DOMAIN_RELEVANCE_FLOOR", 0.15)  # 2026-08-24 — replaces
# the old exact "score == 0" keyword-count check now that domain_score is
# a continuous [0,1] value (real LLM classification, not just a keyword
# tally) — an exact-zero check would almost never trigger against a real
# model's output the way it reliably did against an integer count.
VAGUE_QUERY_CONFIDENCE_FLOOR = _float("VAGUE_QUERY_CONFIDENCE_FLOOR", 0.35)
VAGUE_QUERY_MIN_WORDS = _int("VAGUE_QUERY_MIN_WORDS", 4)

# ── Cache ──
CACHE_SIMILARITY_THRESHOLD = _float("CACHE_SIMILARITY_THRESHOLD", 0.93)
CACHE_DEFAULT_TTL_SECONDS = _int("CACHE_DEFAULT_TTL_SECONDS", 3600 * 24)

# ── Memory ──
MEMORY_DEFAULT_TTL_SECONDS = _int("MEMORY_DEFAULT_TTL_SECONDS", 3600 * 24 * 180)
MEMORY_EMBED_DIMS = _int("MEMORY_EMBED_DIMS", 768)
# Count-based cap, alongside the time-based TTL above — TTL alone does
# nothing for a single heavy-testing day (180 days won't have elapsed),
# so episodic history can still grow unboundedly within one session-heavy
# day. This caps it directly: oldest episodic facts evicted once a user
# crosses this count. Semantic facts (stable, few per user) are NOT
# subject to this cap — only episodic (one per query) actually grows
# unboundedly. See memory/store.py's 2026-08-24 fix for the Mem0
# token-limit failure this was built to prevent.
MEMORY_MAX_EPISODIC_FACTS_PER_USER = _int("MEMORY_MAX_EPISODIC_FACTS_PER_USER", 50)

# ── Chunking ──
CHUNK_TARGET_TOKENS = _int("CHUNK_TARGET_TOKENS", 450)
CHUNK_OVERLAP_TOKENS = _int("CHUNK_OVERLAP_TOKENS", 60)

# ── Gateway / circuit breaker ──
GATEWAY_MAX_RETRIES_PER_PROVIDER = _int("GATEWAY_MAX_RETRIES_PER_PROVIDER", 2)
GATEWAY_MAX_COST_PER_REQUEST = _float("GATEWAY_MAX_COST_PER_REQUEST", 1.00)
CIRCUIT_BREAKER_FAILURE_THRESHOLD = _int("CIRCUIT_BREAKER_FAILURE_THRESHOLD", 3)
CIRCUIT_BREAKER_COOLDOWN_SECONDS = _int("CIRCUIT_BREAKER_COOLDOWN_SECONDS", 30)

# ── Per-provider minimum call interval (new) ──
# Confirmed live (2026-08-24): a SINGLE query against Groq works fine, but
# eval's concurrent batches (multiple calls fired in the same instant)
# reliably blow past Groq's real 8,000 TPM ceiling and trip the shared
# circuit breaker within the first batch — which then locks Groq out for
# the REST of the eval run, making it look like total failure when it was
# really "worked once, then got rate-limited by our own burst." This
# throttle serializes calls to the SAME provider with a minimum gap
# between them, so concurrent eval workers naturally queue instead of
# all hitting Groq in the same instant. 1.5s is conservative for Groq's
# tight free-tier ceiling; providers with looser limits can be given a
# smaller gap via env override if needed.
GATEWAY_MIN_CALL_INTERVAL_SECONDS = _float("GATEWAY_MIN_CALL_INTERVAL_SECONDS", 1.5)

# ── Eval targets (the CI gate) ──
TARGET_RECALL_AT_25 = _float("TARGET_RECALL_AT_25", 0.90)
TARGET_PRECISION_AT_5 = _float("TARGET_PRECISION_AT_5", 0.80)
TARGET_FAITHFULNESS = _float("TARGET_FAITHFULNESS", 0.80)
TARGET_REFUSAL_ACCURACY = _float("TARGET_REFUSAL_ACCURACY", 1.00)

# ── Rate limiting (new) ──
RATE_LIMIT_ENABLED = _bool("RATE_LIMIT_ENABLED", True)
RATE_LIMIT_QUERY_PER_MINUTE = _int("RATE_LIMIT_QUERY_PER_MINUTE", 20)
RATE_LIMIT_INGEST_PER_MINUTE = _int("RATE_LIMIT_INGEST_PER_MINUTE", 5)


def rate_limit_storage_uri() -> str:
    """
    Builds a storage URI for slowapi's Limiter, reusing the same Valkey
    connection info retrieval/cache.py already uses (VALKEY_URI or
    VALKEY_HOST/PORT/PASSWORD) — one Valkey instance backs both the
    semantic cache and rate-limit counters, no separate service needed.
    Falls back to in-memory storage (per-process, not shared across
    workers) if no Valkey config is present at all — fine for a single
    dev process, NOT fine for a multi-worker production deployment, which
    is exactly why this is a function that logs its choice rather than a
    silent default.
    """
    uri = os.environ.get("VALKEY_URI")
    if uri:
        return uri
    host = os.environ.get("VALKEY_HOST")
    if host:
        port = os.environ.get("VALKEY_PORT", "6379")
        password = os.environ.get("VALKEY_PASSWORD")
        auth = f":{password}@" if password else ""
        scheme = "rediss" if _bool("VALKEY_SSL", False) else "redis"
        return f"{scheme}://{auth}{host}:{port}"
    return "memory://"

# ── CORS (new) ──
CORS_ALLOWED_ORIGINS = _list("CORS_ALLOWED_ORIGINS", ["http://localhost:8501"])

# ── Ingestion embedding concurrency (new) ──
# Bounded, not unlimited: parallelizing every chunk's embedding call cuts
# wall-clock ingestion time a lot (25 sequential Gemini calls at ~1s each
# is 25s; 5-way parallel is closer to 5s), but going fully unbounded risks
# bursting past Groq/Gemini free-tier rate limits (see pipeline-parameters.md)
# on a single large document. 5 concurrent workers is a reasonable default
# for the free-tier ceilings already documented there.
INGESTION_EMBED_CONCURRENCY = _int("INGESTION_EMBED_CONCURRENCY", 5)

# ── Eval case concurrency (new) ──
# Same reasoning as ingestion concurrency, applied to eval/test_golden_set.py:
# each golden-set case makes several real API calls (retrieval, Cohere
# rerank, generation, faithfulness judge) — running 26 cases sequentially
# at real-world LLM latency easily exceeds a client's request timeout
# (confirmed live: /admin/eval timing out at 120s from Streamlit). Bounded
# parallelism here, same trade-off as ingestion: faster wall-clock time
# vs. free-tier rate-limit pressure.
EVAL_CONCURRENCY = _int("EVAL_CONCURRENCY", 4)
CONTEXTUALIZATION_MAX_HISTORY_TURNS = _int("CONTEXTUALIZATION_MAX_HISTORY_TURNS", 3)

# ── Observability alerting (new) ──
# Empty ALERT_WEBHOOK_URL disables alerting entirely (no accidental
# webhook spam on a dev machine with nothing configured). Slack-compatible
# JSON payload ({"text": ...}) — also works with Discord and most
# incoming-webhook services unmodified.
ALERT_WEBHOOK_URL = os.environ.get("ALERT_WEBHOOK_URL", "")
# Fires when this many GENERATION calls in a row fell back to the offline
# mock (i.e. every real provider failed for each of them) — this is
# exactly today's real scenario (Groq+Cerebras+Gemini+OpenRouter all down
# at once), the single most actionable "something is actually wrong"
# signal this system can produce on its own, without needing a human
# watching a dashboard.
ALERT_CONSECUTIVE_DEGRADED_THRESHOLD = _int("ALERT_CONSECUTIVE_DEGRADED_THRESHOLD", 3)
# Minimum gap between alerts for the SAME condition — without this, a
# sustained outage (exactly when you'd have this problem) would fire one
# alert per request for the whole duration, which is spam, not an alert.
ALERT_COOLDOWN_SECONDS = _int("ALERT_COOLDOWN_SECONDS", 300)

# ── Audit log (new) ──
AUDIT_LOG_RETENTION_DAYS = _int("AUDIT_LOG_RETENTION_DAYS", 365)
AUDIT_LOG_ROTATION_CHECK_INTERVAL_SECONDS = _int("AUDIT_LOG_ROTATION_CHECK_INTERVAL_SECONDS", 24 * 3600)
# Item 1.7 — real scheduling, not just the manual /admin/audit-log/rotate
# trigger. Honest limitation, stated plainly: this is an in-process
# background thread, which only runs while the process is alive — on
# Render's free tier specifically, the instance SLEEPS after 15 minutes
# of inactivity, so this won't fire reliably on a strict schedule the way
# a real external cron job would. It still provides real value (rotation
# happens automatically whenever the process IS awake and this interval
# has elapsed, rather than requiring someone to remember to call the
# manual endpoint), but a genuinely reliable production schedule needs an
# external scheduler (Render's own Cron Jobs feature on paid tiers, or
# any external cron hitting the manual endpoint) — not something this
# codebase can guarantee from inside itself.

# ── Admin bootstrap (new, 2026-08-25) ──
# Alternative to scripts/create_admin.py's shell-access bootstrap: when
# true (default), the FIRST user ever registered on a fresh deployment
# (zero existing admins) is automatically promoted to admin, no shell
# access required. Every registration AFTER that first one still only
# ever creates the lowest-privilege role — this only fires once, for
# whoever gets there first on a genuinely empty deployment. Set false to
# require scripts/create_admin.py instead, if you'd rather have admin
# creation gated behind actual server access every time, not just "first
# to register."
AUTO_PROMOTE_FIRST_USER_TO_ADMIN = _bool("AUTO_PROMOTE_FIRST_USER_TO_ADMIN", True)

# ── Email delivery (new, 2026-08-25) ──
# Real password-reset email delivery via Resend — replaces the earlier
# dev-mode stopgap (logging the reset token to the server console).
# Empty RESEND_API_KEY falls back to that same console-logging behavior
# automatically — never crashes /auth/forgot-password, same graceful-
# degradation pattern as every other optional integration in this
# codebase. Get a free key at resend.com (no card required, 3,000
# emails/month free, no domain verification needed to start — sending
# from their shared onboarding@resend.dev address works immediately).
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
RESEND_FROM_ADDRESS = os.environ.get("RESEND_FROM_ADDRESS", "onboarding@resend.dev")
# FRONTEND_BASE_URL: needed to build a real clickable reset LINK in the
# email, not just a bare token — set this to wherever Streamlit is
# actually reachable (e.g. your Render frontend URL). Defaults to local
# dev's Streamlit address.
FRONTEND_BASE_URL = os.environ.get("FRONTEND_BASE_URL", "http://localhost:8501")
# When set, auth/store.py, sessions/store.py, and observability/tracer.py
# all use Postgres instead of local SQLite/JSONL files — required for any
# deployment where the filesystem doesn't survive a restart (e.g. Render's
# free tier, which wipes disk on every restart/redeploy with no persistent-
# disk option at that tier). Unset (None) means "keep using local files,"
# which remains fully supported for local development — this was NOT a
# rip-and-replace, both paths coexist.
DATABASE_URL = os.environ.get("DATABASE_URL") or None