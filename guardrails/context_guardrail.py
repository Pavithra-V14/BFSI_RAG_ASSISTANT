"""
Context guardrail — runs AFTER rerank, BEFORE the chunks are handed to the
LLM for generation. This is distinct from both other guardrails:

  input_guardrail.py   checks the QUERY, before retrieval runs
  context_guardrail.py checks the RETRIEVED CHUNKS, before generation runs  <- this file
  output_guardrail.py  checks the GENERATED ANSWER, before the user sees it

Why this layer exists even though ingestion already scans documents:
defense in depth. An obfuscated injection payload can pass the ingestion
scan (different surrounding text, encoding tricks) and only become
dangerous once it lands in a specific retrieved context next to a
specific query. Re-scanning at the narrower point catches what the wider
scan missed, and it's cheap — you're only re-scanning 4-6 chunks, not the
whole corpus.

Five checks, in order (see ADR 0005):
  1. injection re-scan on the actual retrieved text (defense in depth)
  2. access re-verification (defense in depth against a retrieval-layer bug)
  3. staleness check (a superseded chunk must never reach generation)
  4. sufficiency / relevance floor (too little support -> fail closed here,
     before spending a generation call on it)
  5. PII masking on surviving chunk text — separate from output_guardrail's
     mask, which covers the GENERATED answer, not the source material. This
     matters because retrieved chunk text (e.g. from claims data) may
     contain policyholder PII that shouldn't reach a third-party LLM API's
     context window even if the final answer never repeats it.
"""
from __future__ import annotations

import re

import config
from dataclasses import dataclass, field

from ingestion.safety_scan import scan_text
from guardrails.pii import mask_pii

MIN_RELEVANCE_SCORE = config.CONTEXT_MIN_RELEVANCE_SCORE
MIN_SURVIVING_CHUNKS = 1     # fewer than this after filtering -> fail closed


@dataclass
class ContextCheckResult:
    passed: bool
    surviving_chunks: list  # list[retrieval.index.Candidate], filtered
    dropped: list[dict] = field(default_factory=list)  # {chunk_id, reason}
    reason: str | None = None
    pii_masked_chunk_ids: list[str] = field(default_factory=list)
    pii_detected_types: list[str] = field(default_factory=list)  # e.g. ["EMAIL_ADDRESS", "PERSON"] —
    # audit trail of WHAT categories of PII were found, never the actual
    # values — item 7's "log what gets masked" half.


def check_retrieved_context(
    candidates: list,          # list[retrieval.index.Candidate]
    user_role: str,
    min_relevance: float = MIN_RELEVANCE_SCORE,
) -> ContextCheckResult:
    surviving = []
    dropped = []
    pii_masked_ids = []
    all_pii_types = []

    for c in candidates:
        # 1. injection re-scan on the actual chunk text
        scan = scan_text(c.text)
        if not scan.safe:
            dropped.append({"chunk_id": c.chunk_id, "reason": "injection_in_retrieved_content"})
            continue

        # 2. access re-verification (defense in depth — the retrieval-layer
        #    filter should have already enforced this; check again here)
        allowed_roles = c.metadata.get("access_role", ["*"])
        if isinstance(allowed_roles, str):
            allowed_roles = [allowed_roles]
        if "*" not in allowed_roles and user_role not in allowed_roles:
            dropped.append({"chunk_id": c.chunk_id, "reason": "access_scope_violation"})
            continue

        # 3. staleness — a superseded chunk must never reach generation
        if c.metadata.get("effective_to"):
            dropped.append({"chunk_id": c.chunk_id, "reason": "superseded_version"})
            continue

        # 4. relevance floor — below this the chunk is noise, drop it
        #    (this trims candidates below what rerank already ordered,
        #    it doesn't re-order — rerank's ordering is trusted here)
        if c.score < min_relevance:
            dropped.append({"chunk_id": c.chunk_id, "reason": "below_relevance_floor"})
            continue

        # 5. PII mask on the surviving chunk's text before it can reach
        #    generation — mutate a copy, don't touch the original Candidate
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