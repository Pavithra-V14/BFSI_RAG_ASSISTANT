from __future__ import annotations

import re
from dataclasses import dataclass, field


INJECTION_PATTERNS = [
    r"ignore (all|the|any) (previous|prior|above) instructions",
    r"disregard (all|the|any) (previous|prior|above) (instructions|rules)",
    r"you are now (in )?(developer|admin|debug) mode",
    r"system prompt\s*:",
    r"</?(system|assistant|user)>",
    r"reveal (your|the) (system prompt|instructions)",
    r"act as (if you (are|were)|an unrestricted)",
    r"do not (mention|disclose) (this|that) to the user",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]
PDF_ACTIVE_CONTENT_MARKERS = [b"/JavaScript", b"/JS", b"/OpenAction", b"/AA"]

@dataclass
class ScanResult:
    safe: bool
    reasons: list[str] = field(default_factory=list)

def scan_text(text: str) -> ScanResult:
    """Scan extracted text for injection-style instructions."""
    reasons = []
    for pattern in _COMPILED:
        if pattern.search(text):
            reasons.append(f"injection_pattern_matched:{pattern.pattern}")
    return ScanResult(safe=not reasons, reasons=reasons)

def scan_raw_bytes(raw: bytes) -> ScanResult:
    """Scan the raw file bytes for active content before parsing."""
    reasons = []
    for marker in PDF_ACTIVE_CONTENT_MARKERS:
        if marker in raw:
            reasons.append(f"active_content_marker:{marker.decode()}")
    return ScanResult(safe=not reasons, reasons=reasons)

def scan_document(raw_bytes: bytes, extracted_text: str) -> ScanResult:
    """Combined pre-ingestion safety gate. A document must pass both checks."""
    byte_result = scan_raw_bytes(raw_bytes)
    text_result = scan_text(extracted_text)
    reasons = byte_result.reasons + text_result.reasons
    return ScanResult(safe=not reasons, reasons=reasons)
