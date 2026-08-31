from __future__ import annotations

import re

import config
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, field_validator

from guardrails.pii import mask_pii as _shared_mask_pii

NOT_FOUND_MESSAGE = (
    "I couldn't find this in the available policy documents or regulatory "
    "circulars. Please check with a compliance officer, or rephrase if you "
    "think this should be covered."
)

GROUNDEDNESS_THRESHOLD = config.GROUNDEDNESS_THRESHOLD


class CitedAnswer(BaseModel):
    """Guardrails AI / pydantic schema — the structural contract every
    answer must satisfy before it can ship. This is checked independently
    of the groundedness SCORE below; a well-grounded answer that fails this
    schema (e.g. empty citations list on a non-refusal answer) still blocks."""

    text: str = Field(min_length=1)
    citations: list[str] = Field(default_factory=list)
    grounded: bool

    @field_validator("citations")
    @classmethod
    def citations_required_if_grounded(cls, v: list[str], info) -> list[str]:
        grounded = info.data.get("grounded")
        if grounded and not v:
            raise ValueError("grounded=True requires at least one citation")
        return v


@dataclass
class OutputResult:
    text: str
    grounded: bool
    citations: list[str] = field(default_factory=list)
    pii_masked: bool = False
    schema_valid: bool = True
    pii_detected_types: list[str] = field(default_factory=list) 

def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def check_groundedness(answer: str, retrieved_chunks: list[dict]) -> float:
    """
    Lexical-overlap proxy for entailment. See ADR 0008: RAGAS has an
    unresolved transitive dependency conflict with the guardrails-ai/
    langgraph stack in this environment; DeepEval's faithfulness metric
    (eval/run.py) is the production path once a judge model is configured.
    This function is the synchronous, zero-latency floor used inline on
    every request — DeepEval runs offline against the golden set, not
    per-request.
    """
    if not retrieved_chunks:
        return 0.0
    answer_tokens = _tokenize(answer)
    if not answer_tokens:
        return 0.0
    chunk_tokens: set[str] = set()
    for c in retrieved_chunks:
        chunk_tokens |= _tokenize(c.get("text", ""))
    overlap = answer_tokens & chunk_tokens
    return len(overlap) / len(answer_tokens)


def mask_pii(text: str) -> tuple[str, bool]:
    """
    Backward-compatible 2-tuple wrapper — see guardrails/pii.py for the
    real implementation (Presidio-first, regex fallback) and its 3-tuple
    (masked, found, detected_types) signature. This module's own tests
    and OutputResult contract expect a 2-tuple, so that's preserved here;
    apply_output_guardrail() below fetches the 3rd element (detected
    types) directly from the shared function when it needs it for the
    audit trail.
    """
    masked, found, _types = _shared_mask_pii(text)
    return masked, found


def apply_output_guardrail(
    answer: str,
    retrieved_chunks: list[dict],
    include_citations: bool = True,
) -> OutputResult:
    score = check_groundedness(answer, retrieved_chunks)
    if score < GROUNDEDNESS_THRESHOLD or not retrieved_chunks:
        return OutputResult(text=NOT_FOUND_MESSAGE, grounded=False, citations=[])

    masked_text, pii_found, pii_types = _shared_mask_pii(answer)

    citations = []
    if include_citations:
        citations = sorted({
            f"{c.get('source', 'unknown')} — {c.get('chunk_id', 'unknown')}"
            for c in retrieved_chunks
        })

    schema_valid = True
    try:
        CitedAnswer(text=masked_text, citations=citations, grounded=True)
    except Exception:
        schema_valid = False
        return OutputResult(text=NOT_FOUND_MESSAGE, grounded=False, citations=[], schema_valid=False)

    return OutputResult(
        text=masked_text, grounded=True, citations=citations,
        pii_masked=pii_found, schema_valid=schema_valid, pii_detected_types=pii_types,
    )