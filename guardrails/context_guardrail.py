from __future__ import annotations

import re

import config
from dataclasses import dataclass, field

from ingestion.safety_scan import scan_text
from guardrails.pii import mask_pii

MIN_RELEVANCE_SCORE = config.CONTEXT_MIN_RELEVANCE_SCORE
MIN_SURVIVING_CHUNKS = 1    


@dataclass
class ContextCheckResult:
    passed: bool
    surviving_chunks: list  
    dropped: list[dict] = field(default_factory=list) 
    reason: str | None = None
    pii_masked_chunk_ids: list[str] = field(default_factory=list)
    pii_detected_types: list[str] = field(default_factory=list) 
    
def check_retrieved_context(
    candidates: list,         
    user_role: str,
    min_relevance: float = MIN_RELEVANCE_SCORE,
) -> ContextCheckResult:
    surviving = []
    dropped = []
    pii_masked_ids = []
    all_pii_types = []

    for c in candidates:
        scan = scan_text(c.text)
        if not scan.safe:
            dropped.append({"chunk_id": c.chunk_id, "reason": "injection_in_retrieved_content"})
            continue

        allowed_roles = c.metadata.get("access_role", ["*"])
        if isinstance(allowed_roles, str):
            allowed_roles = [allowed_roles]
        if "*" not in allowed_roles and user_role not in allowed_roles:
            dropped.append({"chunk_id": c.chunk_id, "reason": "access_scope_violation"})
            continue

        if c.metadata.get("effective_to"):
            dropped.append({"chunk_id": c.chunk_id, "reason": "superseded_version"})
            continue

        if c.score < min_relevance:
            dropped.append({"chunk_id": c.chunk_id, "reason": "below_relevance_floor"})
            continue

        masked_text, found, types = mask_pii(c.text)
        if found:
            pii_masked_ids.append(c.chunk_id)
            all_pii_types.extend(types)
        c = c.__class__(chunk_id=c.chunk_id, score=c.score, metadata=c.metadata, text=masked_text)

        surviving.append(c)

    if len(surviving) < MIN_SURVIVING_CHUNKS:
        return ContextCheckResult(
            passed=False,
            surviving_chunks=[],
            dropped=dropped,
            reason="insufficient_grounded_context_after_filtering",
        )

    return ContextCheckResult(
        passed=True, surviving_chunks=surviving, dropped=dropped, pii_masked_chunk_ids=pii_masked_ids,
        pii_detected_types=sorted(set(all_pii_types)),
    )