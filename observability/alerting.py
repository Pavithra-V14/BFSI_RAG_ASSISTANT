"""
Observability alerting (item 6) — pushes a Slack-compatible webhook
notification when the system enters a genuinely actionable degraded
state, instead of requiring a human to notice by manually checking
/admin/stats. Everything built before this was pull-only; this is the
first push mechanism.

Primary signal: consecutive generation calls that fell back to the
offline mock (every real provider failed) — this is exactly what
happened today (Groq+Cerebras+Gemini+OpenRouter simultaneously down),
and it's the single clearest "something is actually wrong, not just
one flaky call" signal available without a dedicated scheduler/cron.

Disabled entirely when ALERT_WEBHOOK_URL is unset (config.py) — never
sends anything on a dev machine with nothing configured, and never
crashes a request if the webhook call itself fails (alerting must be
best-effort, not a new point of failure for the actual product).
"""
from __future__ import annotations

import logging
import threading
import time

import config

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_consecutive_degraded = 0
_last_alert_at: dict[str, float] = {}


def _send_webhook(message: str) -> None:
    if not config.ALERT_WEBHOOK_URL:
        return
    try:
        import requests
        requests.post(config.ALERT_WEBHOOK_URL, json={"text": message}, timeout=5)
    except Exception as e:  # noqa: BLE001 — alerting must never break the request it's alerting about
        logger.warning("Alert webhook delivery failed (%s) — alert was NOT sent: %s", str(e)[:200], message)


def _cooldown_ok(alert_key: str) -> bool:
    now = time.time()
    last = _last_alert_at.get(alert_key, 0.0)
    if now - last < config.ALERT_COOLDOWN_SECONDS:
        return False
    _last_alert_at[alert_key] = now
    return True


def record_generation_outcome(degraded: bool, provider: str | None) -> None:
    """
    Call once per real generation attempt (app.py's /query, after the
    gateway call). Tracks a rolling consecutive-degraded counter — resets
    to 0 the moment a real provider succeeds, so a single blip doesn't
    accumulate toward the threshold across an otherwise-healthy period.
    """
    global _consecutive_degraded
    with _lock:
        if degraded:
            _consecutive_degraded += 1
        else:
            _consecutive_degraded = 0
        count = _consecutive_degraded

    if count >= config.ALERT_CONSECUTIVE_DEGRADED_THRESHOLD and _cooldown_ok("consecutive_degraded"):
        _send_webhook(
            f"🚨 BFSI RAG Assistant: {count} consecutive generation calls fell back to the "
            f"offline mock — every configured LLM provider is currently failing. "
            f"Check provider quotas/status (Groq, Gemini, Cerebras, OpenRouter). "
            f"This will keep firing at most once every {config.ALERT_COOLDOWN_SECONDS}s while it persists."
        )


def alert_on_dashboard_thresholds(metrics: dict) -> None:
    """
    Secondary path — called from /admin/stats after computing dashboard
    metrics (observability/dashboard.py), catches conditions the
    per-request counter above wouldn't (e.g. a high fail-closed rate from
    a mix of causes, not just provider outages)."""
    fail_closed_rate = metrics.get("fail_closed_rate")
    if fail_closed_rate is not None and fail_closed_rate > 0.25 and _cooldown_ok("fail_closed_rate"):
        _send_webhook(
            f"⚠️ BFSI RAG Assistant: fail-closed rate is {fail_closed_rate*100:.0f}% "
            f"over the last {metrics.get('request_count', '?')} requests — "
            f"more than 1 in 4 queries are being refused for empty retrieval or "
            f"ungrounded answers. Check corpus coverage and provider health."
        )


def reset_state_for_testing() -> None:
    """Test-only helper — resets module-level counters between test cases
    so they don't bleed into each other via shared process state."""
    global _consecutive_degraded
    with _lock:
        _consecutive_degraded = 0
    _last_alert_at.clear()