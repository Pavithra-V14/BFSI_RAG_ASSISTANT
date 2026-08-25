"""
PII detection and masking — shared by context_guardrail.py (masks PII in
retrieved chunks before they reach generation) and output_guardrail.py
(masks PII in the final answer before it reaches the user).

Two tiers:
  1. Presidio (Microsoft's open-source PII detection library) — real NER-
     based detection via spaCy, catches PII regex fundamentally cannot:
     person names, addresses, PII phrased naturally in a sentence. Needs
     a spaCy language model, downloaded once. Confirmed live (2026-08-24)
     that Presidio degrades gracefully even when its OWN optional network
     calls fail (its URL recognizer tries to fetch a public-suffix list
     for TLD validation via tldextract — failed in a network-restricted
     test here, logged a warning, and every other recognizer still ran
     correctly) — so a restricted network doesn't take the whole thing
     down, just slightly weakens URL validation specifically.
  2. Regex — the original implementation, kept as the fallback when
     Presidio/spaCy aren't installed or fail to load. Catches structured
     patterns (email, phone, PAN, policy number) but not names, addresses,
     or PII phrased in free text — this was flagged early on as "a good
     floor, not a ceiling," and Presidio is what actually raises it.

Both tiers report WHAT TYPE of PII was found and WHERE (never the PII
VALUE itself) via detected_types, for audit logging (item 7's other half)
— see guardrails/context_guardrail.py and output_guardrail.py's callers.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ── Tier 2: regex (general PII fallback) + domain-specific patterns (ALWAYS run) ──
PII_PATTERNS = {
    "email": re.compile(r"[\w\.-]+@[\w\.-]+\.\w+"),
    "phone": re.compile(r"\b\d{10}\b"),
    "pan": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
}

# Domain-specific patterns with no general-purpose PII-detector equivalent
# (Presidio has no concept of "insurance policy number") — these run
# UNCONDITIONALLY, regardless of which general-PII tier (Presidio or
# regex) handles email/phone/name detection. Confirmed live: an earlier
# version of this module only ran these when Presidio was UNAVAILABLE,
# which meant switching to Presidio silently stopped masking policy
# numbers — a real regression, caught by testing before it shipped.
DOMAIN_SPECIFIC_PATTERNS = {
    "policy_number": re.compile(r"\bPOL[-/]?\d{6,}\b", re.IGNORECASE),
}

# 2026-08-24 — added after a real, confirmed gap: Presidio's PERSON
# recognizer (backed by a spaCy NER model) returned ZERO candidates for
# "Pavithra V" — not a low-confidence miss, genuinely no match at all.
#
# 2026-08-25 update, more precise: initially assumed this was a small-
# model-vs-large-model accuracy gap (en_core_web_sm vs en_core_web_lg).
# Tested directly after upgrading to en_core_web_lg — the LARGER model
# fails on "Pavithra V" identically (zero candidates), while both models
# correctly catch full two-part names regardless of origin ("Rajesh
# Kumar", "Priya Sharma", "John Smith" all score 0.85 on either model).
# So this is NOT a general small-vs-large or Western-vs-non-Western
# accuracy gap — model size doesn't fix it. The actual, narrower cause:
# neither spaCy model's NER training data represents the "FirstName
# LastInitial" abbreviated-surname format (a naming convention common in
# South India) as a name-shaped span at all. Upgrading the model is still
# worthwhile for its broader accuracy gains, but won't close THIS specific
# gap — which is exactly why this format-based (not NER-based) pattern
# exists as a permanent supplement, not a stopgap to remove later.
# Model-independent: it works the same regardless of whose name it is,
# and regardless of which spaCy model is installed.
NAME_INTRODUCTION_PATTERN = re.compile(
    r"\b(?i:my name is|i am|i'm|this is)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]*\.?){0,3})",
)


def _mask_name_introductions(text: str) -> tuple[str, bool, list[str]]:
    if not NAME_INTRODUCTION_PATTERN.search(text):
        return text, False, []
    masked = NAME_INTRODUCTION_PATTERN.sub(
        lambda m: m.group(0)[: m.start(1) - m.start(0)] + "[REDACTED_PERSON]", text
    )
    return masked, True, ["name_introduction"]


def _mask_domain_specific(text: str) -> tuple[str, bool, list[str]]:
    masked, found = text, False
    detected_types: list[str] = []
    for label, pattern in DOMAIN_SPECIFIC_PATTERNS.items():
        if pattern.search(masked):
            found = True
            detected_types.append(label)
            masked = pattern.sub(f"[REDACTED_{label.upper()}]", masked)
    return masked, found, detected_types


def _mask_pii_regex(text: str) -> tuple[str, bool, list[str]]:
    masked, found = text, False
    detected_types: list[str] = []
    for label, pattern in PII_PATTERNS.items():
        if pattern.search(masked):
            found = True
            detected_types.append(label)
            masked = pattern.sub(f"[REDACTED_{label.upper()}]", masked)
    return masked, found, detected_types


# ── Tier 1: Presidio ──

def _get_presidio_engine():
    """Lazily loads Presidio's AnalyzerEngine once per process. Returns
    None (triggering the regex fallback) if presidio-analyzer isn't
    installed or the spaCy model can't be loaded — same graceful-
    degradation pattern as every other optional dependency in this
    codebase (LLM Guard, the local reranker, etc.)."""
    if not hasattr(_get_presidio_engine, "_cached"):
        try:
            from presidio_analyzer import AnalyzerEngine
            _get_presidio_engine._cached = AnalyzerEngine()
        except Exception as e:  # noqa: BLE001 — install/model-load can fail many ways
            logger.warning(
                "Presidio unavailable (%s) — falling back to regex PII masking. "
                "Install with: pip install presidio-analyzer, then download a "
                "spaCy model (e.g. en_core_web_lg).", str(e)[:200],
            )
            _get_presidio_engine._cached = None
    return _get_presidio_engine._cached


# Presidio entity types worth masking for a BFSI compliance context —
# deliberately narrower than Presidio's full default set (which also
# flags things like IP addresses, crypto wallet addresses, IBANs — not
# relevant noise for this domain, and masking too aggressively would
# degrade legitimate policy-clause text unnecessarily).
_PRESIDIO_ENTITIES = [
    "EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON", "LOCATION",
    "CREDIT_CARD", "US_SSN", "IN_PAN", "IN_AADHAAR",
]


def _mask_pii_presidio(text: str, engine) -> tuple[str, bool, list[str]]:
    results = engine.analyze(text=text, language="en", entities=_PRESIDIO_ENTITIES)
    if not results:
        return text, False, []
    # Mask longest-match-first so overlapping spans don't corrupt offsets
    # when substituting — same reasoning as any multi-span text redaction.
    results = sorted(results, key=lambda r: r.start, reverse=True)
    masked = text
    detected_types: list[str] = []
    for r in results:
        if r.score < 0.5:  # low-confidence detections create more noise than
            continue        # value in a compliance document full of legitimate
                             # clause numbers, dates, and policy terminology
        detected_types.append(r.entity_type)
        masked = masked[:r.start] + f"[REDACTED_{r.entity_type}]" + masked[r.end:]
    return masked, bool(detected_types), detected_types


def mask_pii(text: str) -> tuple[str, bool, list[str]]:
    """
    Returns (masked_text, pii_found, detected_types). detected_types is
    the list of PII categories found (e.g. ["EMAIL_ADDRESS", "PERSON"]) —
    never the actual PII values — for audit logging without creating a
    second copy of the sensitive data itself.

    Domain-specific patterns (policy numbers) run UNCONDITIONALLY first,
    then whichever general-PII tier is available (Presidio preferred,
    regex fallback) — see DOMAIN_SPECIFIC_PATTERNS' comment for why these
    can't just be "the fallback path."
    """
    text, domain_found, domain_types = _mask_domain_specific(text)
    text, name_intro_found, name_intro_types = _mask_name_introductions(text)

    engine = _get_presidio_engine()
    if engine is not None:
        try:
            text, general_found, general_types = _mask_pii_presidio(text, engine)
        except Exception as e:  # noqa: BLE001 — a single failed analysis must not crash the request
            logger.warning("Presidio analysis failed (%s), falling back to regex for this call", str(e)[:200])
            text, general_found, general_types = _mask_pii_regex(text)
    else:
        text, general_found, general_types = _mask_pii_regex(text)

    return text, (domain_found or name_intro_found or general_found), domain_types + name_intro_types + general_types