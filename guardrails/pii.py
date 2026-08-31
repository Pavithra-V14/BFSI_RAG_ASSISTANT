from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)
PII_PATTERNS = {
    "email": re.compile(r"[\w\.-]+@[\w\.-]+\.\w+"),
    "phone": re.compile(r"\b\d{10}\b"),
    "pan": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
}

DOMAIN_SPECIFIC_PATTERNS = {
    "policy_number": re.compile(r"\bPOL[-/]?\d{6,}\b", re.IGNORECASE),
}

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
        except Exception as e: 
            logger.warning(
                "Presidio unavailable (%s) — falling back to regex PII masking. "
                "Install with: pip install presidio-analyzer, then download a "
                "spaCy model (e.g. en_core_web_lg).", str(e)[:200],
            )
            _get_presidio_engine._cached = None
    return _get_presidio_engine._cached


_PRESIDIO_ENTITIES = [
    "EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON", "LOCATION",
    "CREDIT_CARD", "US_SSN", "IN_PAN", "IN_AADHAAR",
]

def _mask_pii_presidio(text: str, engine) -> tuple[str, bool, list[str]]:
    results = engine.analyze(text=text, language="en", entities=_PRESIDIO_ENTITIES)
    if not results:
        return text, False, []
    results = sorted(results, key=lambda r: r.start, reverse=True)
    masked = text
    detected_types: list[str] = []
    for r in results:
        if r.score < 0.5: 
            continue      
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
        except Exception as e:  
            logger.warning("Presidio analysis failed (%s), falling back to regex for this call", str(e)[:200])
            text, general_found, general_types = _mask_pii_regex(text)
    else:
        text, general_found, general_types = _mask_pii_regex(text)

    return text, (domain_found or name_intro_found or general_found), domain_types + name_intro_types + general_types