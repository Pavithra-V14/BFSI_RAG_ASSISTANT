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
DOMAIN_RELEVANCE_FLOOR = _float("DOMAIN_RELEVANCE_FLOOR", 0.15) 

VAGUE_QUERY_CONFIDENCE_FLOOR = _float("VAGUE_QUERY_CONFIDENCE_FLOOR", 0.35)
VAGUE_QUERY_MIN_WORDS = _int("VAGUE_QUERY_MIN_WORDS", 4)

# ── Cache ──
CACHE_SIMILARITY_THRESHOLD = _float("CACHE_SIMILARITY_THRESHOLD", 0.93)
CACHE_DEFAULT_TTL_SECONDS = _int("CACHE_DEFAULT_TTL_SECONDS", 3600 * 24)

# ── Memory ──
MEMORY_DEFAULT_TTL_SECONDS = _int("MEMORY_DEFAULT_TTL_SECONDS", 3600 * 24 * 180)
MEMORY_EMBED_DIMS = _int("MEMORY_EMBED_DIMS", 768)
MEMORY_MAX_EPISODIC_FACTS_PER_USER = _int("MEMORY_MAX_EPISODIC_FACTS_PER_USER", 50)

# ── Chunking ──
CHUNK_TARGET_TOKENS = _int("CHUNK_TARGET_TOKENS", 450)
CHUNK_OVERLAP_TOKENS = _int("CHUNK_OVERLAP_TOKENS", 60)

# ── Gateway / circuit breaker ──
GATEWAY_MAX_RETRIES_PER_PROVIDER = _int("GATEWAY_MAX_RETRIES_PER_PROVIDER", 2)
GATEWAY_MAX_COST_PER_REQUEST = _float("GATEWAY_MAX_COST_PER_REQUEST", 1.00)
CIRCUIT_BREAKER_FAILURE_THRESHOLD = _int("CIRCUIT_BREAKER_FAILURE_THRESHOLD", 3)
CIRCUIT_BREAKER_COOLDOWN_SECONDS = _int("CIRCUIT_BREAKER_COOLDOWN_SECONDS", 30)

# ── Per-provider minimum call interval ──
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

# ── Ingestion embedding concurrency ──
INGESTION_EMBED_CONCURRENCY = _int("INGESTION_EMBED_CONCURRENCY", 5)

# ── Eval case concurrency ──
EVAL_CONCURRENCY = _int("EVAL_CONCURRENCY", 4)
CONTEXTUALIZATION_MAX_HISTORY_TURNS = _int("CONTEXTUALIZATION_MAX_HISTORY_TURNS", 3)

# ── Observability alerting ──
ALERT_WEBHOOK_URL = os.environ.get("ALERT_WEBHOOK_URL", "")
ALERT_CONSECUTIVE_DEGRADED_THRESHOLD = _int("ALERT_CONSECUTIVE_DEGRADED_THRESHOLD", 3)
ALERT_COOLDOWN_SECONDS = _int("ALERT_COOLDOWN_SECONDS", 300)

# ── Audit log ──
AUDIT_LOG_RETENTION_DAYS = _int("AUDIT_LOG_RETENTION_DAYS", 365)
AUDIT_LOG_ROTATION_CHECK_INTERVAL_SECONDS = _int("AUDIT_LOG_ROTATION_CHECK_INTERVAL_SECONDS", 24 * 3600)

# ── Admin bootstrap 
AUTO_PROMOTE_FIRST_USER_TO_ADMIN = _bool("AUTO_PROMOTE_FIRST_USER_TO_ADMIN", True)

# ── Email delivery 
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
RESEND_FROM_ADDRESS = os.environ.get("RESEND_FROM_ADDRESS", "onboarding@resend.dev")
FRONTEND_BASE_URL = os.environ.get("FRONTEND_BASE_URL", "http://localhost:8501")
DATABASE_URL = os.environ.get("DATABASE_URL") or None