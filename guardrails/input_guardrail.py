"""
Input guardrail — runs BEFORE cache/retrieval touch the query (see ADR 0005).

PRIMARY: LLM Guard's PromptInjection scanner (ML-based, downloads a model on
first use). FALLBACK: regex pattern check, used automatically if the model
can't be loaded (e.g. no network access to the model hub in this environment)
— logs a warning so the gap is visible, not silent. In a deployment with
model-hub access, LLM Guard's scanner engages automatically, no code change
needed.
"""
from __future__ import annotations

import logging

import config
import re
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)

_REGEX_INJECTION_PATTERNS = [
    r"ignore (all|the|any) (previous|prior|above) instructions",
    r"disregard (all|the|any) (previous|prior|above) (instructions|rules)",
    r"you are now (in )?(developer|admin|debug) mode",
    r"system prompt\s*:",
    r"</?(system|assistant|user)>",
    r"reveal (your|the) (system prompt|instructions)",
    r"act as (if you (are|were)|an unrestricted)",
    r"do not (mention|disclose) (this|that) to the user",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in _REGEX_INJECTION_PATTERNS]

PDF_ACTIVE_CONTENT_MARKERS = [b"/JavaScript", b"/JS", b"/OpenAction", b"/AA"]

DOMAIN_KEYWORDS = {
    "policy", "claim", "insurance", "insurer", "insured", "premium",
    "hospitalization", "hospital", "waiting", "period", "exclusion",
    "portability", "coverage", "sum insured", "copayment", "co-payment",
    "grievance", "ombudsman", "kyc", "circular", "regulation", "irdai",
    "rbi", "compliance", "reimbursement", "cashless", "settlement",
    "underwriting", "renewal", "policyholder", "endorsement",
}

REFUSAL_MESSAGE = (
    "I'm scoped to answer questions about our BFSI compliance and claims "
    "policy documents — I can't help with general questions outside that "
    "domain. Ask me about a policy clause, claim process, or regulatory "
    "requirement and I'll do my best."
)


@dataclass
class ScanResult:
    safe: bool
    reasons: list[str]


@lru_cache(maxsize=1)
def _get_llm_guard_scanner():
    """Lazily loads LLM Guard's ML-based PromptInjection scanner. Returns
    None if the model can't be fetched (network-restricted environment),
    or if the ML scanner is explicitly disabled (see below), or if a call
    fails for any other reason — triggering the regex fallback everywhere
    this is called.

    DISABLE_LLM_GUARD_ML=true — set this to skip the ML scanner entirely,
    even when the model hub IS reachable. Added for a real, confirmed
    case: on a machine where huggingface.co is reachable, the model
    downloads and constructs successfully, but LLM Guard's own internal
    pipeline re-initializes the model on EVERY .scan() call (not just
    once at construction — confirmed live via repeated "Initialized
    classification model" log lines on every single request), and on
    some torch/transformers version combinations that per-call
    re-initialization fails with a meta-tensor device-placement error.
    Either way — slow-but-working or fast-failing — the regex fallback
    already provides equivalent protection for the injection patterns
    this app actually needs to catch, so this toggle exists to skip the
    repeated cost/failure entirely rather than pay it on every request.
    """
    import os

    if os.environ.get("DISABLE_LLM_GUARD_ML", "false").lower() in ("true", "1", "yes"):
        return None

    import socket
    try:
        socket.setdefaulttimeout(1.5)
        socket.gethostbyname("huggingface.co")
    except OSError:
        logger.warning(
            "huggingface.co unreachable — skipping LLM Guard ML scanner load, "
            "using regex injection check. Expected in network-restricted "
            "environments; no code change needed once model-hub access is available."
        )
        return None

    try:
        from llm_guard.input_scanners import PromptInjection
        return PromptInjection()
    except Exception as e:  # noqa: BLE001 — model download / init can fail many ways
        logger.warning(
            "LLM Guard PromptInjection scanner unavailable (%s) — "
            "falling back to regex injection check.", str(e)[:200],
        )
        return None


def scan_text(text: str) -> ScanResult:
    scanner = _get_llm_guard_scanner()
    if scanner is not None:
        try:
            _, is_valid, _ = scanner.scan(text)
            if not is_valid:
                return ScanResult(safe=False, reasons=["llm_guard:prompt_injection_detected"])
            return ScanResult(safe=True, reasons=[])
        except Exception as e:  # noqa: BLE001
            logger.warning("LLM Guard scan failed (%s), falling back to regex", str(e)[:200])
    reasons = [f"regex:injection_pattern_matched:{p.pattern}" for p in _COMPILED if p.search(text)]
    return ScanResult(safe=not reasons, reasons=reasons)


def scan_raw_bytes(raw: bytes) -> ScanResult:
    reasons = [f"active_content_marker:{m.decode()}" for m in PDF_ACTIVE_CONTENT_MARKERS if m in raw]
    return ScanResult(safe=not reasons, reasons=reasons)


@dataclass
class GuardrailDecision:
    action: str          # "proceed" | "rewrite" | "refuse" | "block"
    query: str
    reason: str | None = None


def _domain_score_keyword(query: str) -> float:
    """
    Fallback tier — a keyword-count heuristic normalized to [0, 1]. This
    was the ONLY tier until 2026-08-24; kept as the fallback for when no
    LLM provider is configured or the classification call fails. Genuinely
    crude: catches zero domain keywords for perfectly legitimate questions
    phrased without any of the listed words, and can't distinguish "this
    query is ABOUT insurance" from "this query happens to contain the
    word 'claim'" the way real classification can.
    """
    lowered = query.lower()
    count = sum(1 for kw in DOMAIN_KEYWORDS if kw in lowered)
    return min(count / 2, 1.0)  # 2+ keyword hits = fully in-domain,
    # 0 hits = 0.0 (preserves the original "score == 0 -> possibly out of
    # domain" behavior when this is the active tier)


def _domain_score_llm(query: str) -> float | None:
    """
    Primary tier (item 10, 2026-08-24) — a real classification call
    instead of keyword counting. Returns None (triggering the keyword
    fallback) when no provider is configured, the call fails, or the
    response can't be parsed as a number — never lets a malformed
    response silently corrupt the domain-relevance decision. Uses the
    shared gateway singleton (get_gateway()) — NOT a fresh LLMGateway()
    instance, per the circuit-breaker sharing bug found and fixed
    elsewhere in this codebase on 2026-08-24; a fresh instance here would
    have its own empty breaker and reintroduce that exact class of bug.
    """
    import os

    if not (os.environ.get("GROQ_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        return None

    try:
        from gateway.llm_gateway import get_gateway
        gateway = get_gateway()
        system_prompt = (
            "Rate how relevant this query is to BFSI (banking, financial "
            "services, insurance) compliance, claims, or regulatory topics — "
            "insurance policies, claim processes, KYC, regulatory circulars, "
            "grievances, premiums, coverage, and similar. Respond with ONLY a "
            "number from 0.0 (completely unrelated, e.g. cooking, sports, "
            "general trivia) to 1.0 (clearly about BFSI compliance/claims/"
            "insurance). No words, no explanation — just the number."
        )
        result = gateway.generate(system_prompt=system_prompt, context_chunks=[], query=query)
        from observability.tracer import log_gateway_attempt
        log_gateway_attempt("domain_relevance_classification", result)
        if result["provider"] in ("offline_mock", "offline_mock_fallback", None):
            return None  # mock generator can't meaningfully score relevance
        score = float(result["text"].strip())
        return max(0.0, min(score, 1.0))  # clamp — a malformed "1.5" or
        # negative number from the model shouldn't silently break the
        # downstream threshold comparison
    except (ValueError, TypeError):
        logger.warning("Domain relevance LLM response wasn't a parseable number, using keyword fallback")
        return None
    except Exception as e:  # noqa: BLE001 — classification must never block a query
        logger.warning("Domain relevance LLM call failed (%s), using keyword fallback", str(e)[:200])
        return None


def _domain_score(query: str) -> float:
    """Real classification when a provider is configured, keyword
    fallback otherwise — see the two functions above."""
    llm_score = _domain_score_llm(query)
    return llm_score if llm_score is not None else _domain_score_keyword(query)


def check_input(query: str, retrieval_confidence: float | None = None) -> GuardrailDecision:
    scan = scan_text(query)
    if not scan.safe:
        return GuardrailDecision(action="block", query=query, reason="injection_detected")

    score = _domain_score(query)
    if score < config.DOMAIN_RELEVANCE_FLOOR and (retrieval_confidence is None or retrieval_confidence < config.OUT_OF_DOMAIN_CONFIDENCE_FLOOR):
        return GuardrailDecision(action="refuse", query=query, reason="out_of_domain")

    if len(query.split()) < config.VAGUE_QUERY_MIN_WORDS or (retrieval_confidence is not None and retrieval_confidence < config.VAGUE_QUERY_CONFIDENCE_FLOOR):
        rewritten = _rewrite_vague_query(query)
        return GuardrailDecision(action="rewrite", query=rewritten, reason="low_confidence_in_domain")

    return GuardrailDecision(action="proceed", query=query)


def _rewrite_vague_query(query: str) -> str:
    """Production: call the LLM gateway with a rewrite prompt (optionally
    HyDE-style). Demo-safe stand-in: expand with domain framing."""
    return f"BFSI insurance policy or regulatory question: {query.strip()}"