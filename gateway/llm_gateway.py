from __future__ import annotations

import logging

import config
import os
import re
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)
_OPENROUTER_DEFAULT_MODEL = "openai/gpt-oss-20b:free" 

PROVIDER_CONFIG = {
    "groq": {"model": "groq/openai/gpt-oss-120b", "env_key": "GROQ_API_KEY"},
    "gemini": {"model": "gemini/gemini-2.5-flash", "env_key": "GEMINI_API_KEY"},
    "openrouter": {
        "model": f"openrouter/{os.environ.get('OPENROUTER_MODEL', _OPENROUTER_DEFAULT_MODEL)}",
        "env_key": "OPENROUTER_API_KEY",
    },
   
}


@dataclass
class CircuitBreakerState:
    failure_count: int = 0
    opened_at: float | None = None
    failure_threshold: int = config.CIRCUIT_BREAKER_FAILURE_THRESHOLD
    cooldown_seconds: int = config.CIRCUIT_BREAKER_COOLDOWN_SECONDS

    def record_failure(self) -> None:
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.opened_at = time.time()

    def record_success(self) -> None:
        self.failure_count = 0
        self.opened_at = None

    def is_open(self) -> bool:
        if self.opened_at is None:
            return False
        if time.time() - self.opened_at > self.cooldown_seconds:
            self.opened_at = None
            self.failure_count = 0
            return False
        return True


@dataclass
class LLMGateway:
    providers: list[str] = field(default_factory=lambda: ["groq","gemini", "openrouter"])
    max_retries_per_provider: int = config.GATEWAY_MAX_RETRIES_PER_PROVIDER
    max_cost_per_request: float = config.GATEWAY_MAX_COST_PER_REQUEST
    _breakers: dict[str, CircuitBreakerState] = field(default_factory=dict)
    _last_call_at: dict[str, float] = field(default_factory=dict)
    _pacing_lock: threading.Lock = field(default_factory=threading.Lock)

    def _breaker(self, provider: str) -> CircuitBreakerState:
        return self._breakers.setdefault(provider, CircuitBreakerState())

    def _wait_for_pacing_slot(self, provider: str) -> None:
        """
        Serializes calls to the SAME provider with a minimum gap between
        them (config.GATEWAY_MIN_CALL_INTERVAL_SECONDS) — see the config
        comment for why this exists: concurrent eval workers (or any
        concurrent callers sharing this gateway singleton) firing at the
        exact same instant is what actually trips a tight free-tier limit
        like Groq's, not sustained real usage. This makes concurrent
        callers queue naturally instead of bursting, without any caller
        needing to know or care that pacing is happening.
        """
        with self._pacing_lock:
            last = self._last_call_at.get(provider, 0.0)
            now = time.time()
            wait = config.GATEWAY_MIN_CALL_INTERVAL_SECONDS - (now - last)
            if wait > 0:
                time.sleep(wait)
            self._last_call_at[provider] = time.time()

    def _configured_providers(self) -> list[str]:
        return [
            p for p in self.providers
            if os.environ.get(PROVIDER_CONFIG[p]["env_key"])
        ]

    def generate(self, system_prompt: str, context_chunks: list[str], query: str) -> dict:
        estimated_cost = self._estimate_cost(system_prompt, context_chunks, query)
        provider_attempts: list[dict] = [] 
        if estimated_cost > self.max_cost_per_request:
            return {"text": "", "error": "circuit_breaker:max_cost_exceeded", "provider": None,
                    "estimated_cost": estimated_cost, "provider_attempts": provider_attempts}

        configured = self._configured_providers()
        if not configured:
            logger.info("No provider API keys configured — using offline mock generator")
            return {
                "text": self._mock_generate(context_chunks, query),
                "error": None, "provider": "offline_mock", "attempt": 1,
                "estimated_cost": 0.0, "provider_attempts": provider_attempts,
            }

        last_error = None
        for provider in configured:
            breaker = self._breaker(provider)
            if breaker.is_open():
                last_error = f"circuit_open:{provider}"
                provider_attempts.append({"provider": provider, "outcome": "circuit_open"})
                continue
            for attempt in range(self.max_retries_per_provider):
                self._wait_for_pacing_slot(provider)
                try:
                    text = self._call_provider(provider, system_prompt, context_chunks, query)
                    breaker.record_success()
                    provider_attempts.append({"provider": provider, "outcome": "success"})
                    return {
                        "text": text, "error": None, "provider": provider, "attempt": attempt + 1,
                        "estimated_cost": estimated_cost, "provider_attempts": provider_attempts,
                    }
                except Exception as e: 
                    err_str = str(e)
                    is_rate_limit = "rate" in err_str.lower() or "quota" in err_str.lower() or "429" in err_str
                    breaker.record_failure()
                    last_error = f"{'rate_limit' if is_rate_limit else 'error'}:{provider}:{err_str[:200]}"
                    provider_attempts.append({
                        "provider": provider,
                        "outcome": "rate_limited" if is_rate_limit else "error",
                    })
                    logger.warning("Provider %s failed (attempt %d): %s", provider, attempt + 1, err_str[:200])

        logger.warning("All configured providers exhausted, falling back to offline mock: %s", last_error)
        return {
            "text": self._mock_generate(context_chunks, query),
            "error": f"all_providers_exhausted:{last_error}",
            "provider": "offline_mock_fallback",
            "estimated_cost": 0.0,
            "provider_attempts": provider_attempts,
        }

    def _call_provider(self, provider: str, system_prompt: str, context_chunks: list[str], query: str) -> str:
        import litellm

        context_block = "\n\n".join(f"[Chunk {i+1}] {c}" for i, c in enumerate(context_chunks))
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {query}"},
        ]
        response = litellm.completion(
            model=PROVIDER_CONFIG[provider]["model"],
            messages=messages,
            timeout=15,
            max_tokens=400,
        )
        return response.choices[0].message.content

    @staticmethod
    def _mock_generate(context_chunks: list[str], query: str) -> str:
        """Deterministic offline stand-in: returns the sentence in the
        context most lexically similar to the query. Proves grounding
        flows through correctly without a real model call."""
        q_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        best_sentence, best_score = "", -1.0
        for chunk in context_chunks:
            for sentence in re.split(r"(?<=[.!?])\s+", chunk):
                s_tokens = set(re.findall(r"[a-z0-9]+", sentence.lower()))
                if not s_tokens:
                    continue
                score = len(q_tokens & s_tokens) / len(s_tokens)
                if score > best_score:
                    best_score, best_sentence = score, sentence.strip()
        return best_sentence or "No relevant information found in the provided context."

    @staticmethod
    def _estimate_cost(system_prompt: str, context_chunks: list[str], query: str) -> float:
        total_chars = len(system_prompt) + sum(len(c) for c in context_chunks) + len(query)
        approx_tokens = total_chars / 4
        return round((approx_tokens / 1000) * 0.01, 4) 


# ─────────────────────────── shared singleton ───────────────────────────

_shared_gateway: LLMGateway | None = None

def get_gateway() -> LLMGateway:
    global _shared_gateway
    if _shared_gateway is None:
        _shared_gateway = LLMGateway()
    return _shared_gateway