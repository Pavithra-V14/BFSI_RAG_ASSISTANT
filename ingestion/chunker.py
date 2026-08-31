from __future__ import annotations

import hashlib
import logging

import config
import re
from dataclasses import dataclass, field
from datetime import date

from gateway.llm_gateway import get_gateway

logger = logging.getLogger(__name__)

CLAUSE_BOUNDARY = re.compile(
    r"^(Section\s+\d+[\.\)]|Clause\s+\d+[\.\)]|\d+\.\d+)\s", re.MULTILINE
)

TARGET_TOKENS = config.CHUNK_TARGET_TOKENS         
OVERLAP_TOKENS = config.CHUNK_OVERLAP_TOKENS        
CHARS_PER_TOKEN = 4        

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    source: str
    effective_date: str
    product_line: str
    clause_type: str
    version: int
    content_hash: str
    access_role: list[str] = field(default_factory=lambda: ["*"])
    effective_to: str | None = None


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def multimodal_chunks(
    tables_as_markdown: list[str],
    image_captions: list[str],
    doc_id: str,
    source: str,
    product_line: str,
    start_seq: int,
    version: int = 1,
    effective_date: str | None = None,
) -> list[Chunk]:
    """
    Builds Chunk objects for extracted tables/image-captions — clause_type
    is set directly ("table"/"image_caption"), not run through
    _classify_clause(), since we already know exactly what these are; the
    classifier exists to categorize ambiguous prose text, not structured
    extraction output. start_seq continues the chunk_id numbering from
    wherever clause_chunk() left off, so chunk_ids stay unique and
    sequential within a document regardless of content type.
    """
    effective_date = effective_date or date.today().isoformat()
    chunks: list[Chunk] = []
    seq = start_seq

    for table_md in tables_as_markdown:
        seq += 1
        chunks.append(Chunk(
            chunk_id=f"{doc_id}::c{seq:03d}",
            doc_id=doc_id, text=table_md, source=source,
            effective_date=effective_date, product_line=product_line,
            clause_type="table", version=version,
            content_hash=_content_hash(table_md),
        ))

    for caption in image_captions:
        seq += 1
        chunks.append(Chunk(
            chunk_id=f"{doc_id}::c{seq:03d}",
            doc_id=doc_id, text=caption, source=source,
            effective_date=effective_date, product_line=product_line,
            clause_type="image_caption", version=version,
            content_hash=_content_hash(caption),
        ))

    return chunks


def _split_oversized(text: str, target: int, overlap: int) -> list[str]:
    """Only called when a single clause exceeds TARGET_TOKENS chars-worth."""
    max_chars = target * CHARS_PER_TOKEN
    overlap_chars = overlap * CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return [text]
    parts, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        parts.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap_chars
    return parts


def clause_chunk(
    text: str,
    doc_id: str,
    source: str,
    product_line: str,
    version: int = 1,
    effective_date: str | None = None,
) -> list[Chunk]:
    """Split raw document text into clause-bounded Chunk objects."""
    effective_date = effective_date or date.today().isoformat()

    boundaries = [m.start() for m in CLAUSE_BOUNDARY.finditer(text)]
    if not boundaries:
        
        raw_clauses = [p for p in text.split("\n\n") if p.strip()]
    else:
        boundaries.append(len(text))
        raw_clauses = [
            text[boundaries[i] : boundaries[i + 1]].strip()
            for i in range(len(boundaries) - 1)
        ]

    chunks: list[Chunk] = []
    seq = 0
    for clause_text in raw_clauses:
        if not clause_text.strip():
            continue
        clause_type = _classify_clause(clause_text)
        for piece in _split_oversized(clause_text, TARGET_TOKENS, OVERLAP_TOKENS):
            seq += 1
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::c{seq:03d}",
                    doc_id=doc_id,
                    text=piece.strip(),
                    source=source,
                    effective_date=effective_date,
                    product_line=product_line,
                    clause_type=clause_type,
                    version=version,
                    content_hash=_content_hash(piece),
                )
            )
    return chunks


CLAUSE_CATEGORIES = [
    "exclusion", "waiting_period", "kyc", "claims",
    "portability", "copayment", "grievance", "general",
]


def _classify_clause_llm(text: str) -> str | None:
    """
    Real classification via a short LLM call — one category label out,
    for the same set of categories the keyword version used. Returns None
    (triggering the keyword fallback below) when no real provider is
    configured, the call fails, or the LLM's answer isn't one of the
    known categories — never lets a malformed response silently mislabel
    a chunk.

    Cost/latency trade-off, stated explicitly: this is one LLM call per
    clause at ingestion time (e.g. ~25 calls for a 7-section policy
    document) — negligible for occasional document ingestion, worth
    knowing about before ingesting a very large corpus in one run. Groq's
    latency (see gateway/llm_gateway.py) keeps this fast in practice; the
    real cost is API call volume against free-tier rate limits, not
    wall-clock time.
    """
    import os

    if not (os.environ.get("GROQ_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        return None

    try:
        from gateway.llm_gateway import get_gateway
        gateway = get_gateway()
        system_prompt = (
            "Classify the following insurance policy or regulatory clause into "
            f"EXACTLY ONE of these categories: {', '.join(CLAUSE_CATEGORIES)}. "
            "Respond with ONLY the category name, lowercase, nothing else."
        )
        result = gateway.generate(system_prompt=system_prompt, context_chunks=[], query=text)
        if result["provider"] in ("offline_mock", "offline_mock_fallback", None):
            return None
        label = result["text"].strip().lower().strip('."\'')
        return label if label in CLAUSE_CATEGORIES else None
    except Exception as e:  
        logger.warning("LLM clause classification failed (%s), using keyword fallback", str(e)[:200])
        return None


def _classify_clause_keyword(text: str) -> str:
    """
    Keyword fallback — used when no LLM provider is configured or the
    LLM call fails. Preserved (not deleted) because it's what keeps
    ingestion working offline, same graceful-degradation pattern as
    every other module in this codebase.
    """
    lowered = text.lower()
    if "exclu" in lowered:
        return "exclusion"
    if "waiting period" in lowered:
        return "waiting_period"
    if "kyc" in lowered or "identity" in lowered:
        return "kyc"
    if "claim" in lowered:
        return "claims"
    if "porta" in lowered or "migrat" in lowered:
        return "portability"
    if "co-payment" in lowered or "copay" in lowered:
        return "copayment"
    if "grievance" in lowered or "ombudsman" in lowered:
        return "grievance"
    return "general"


def _classify_clause(text: str) -> str:
    """Real classification when a provider is configured, keyword
    fallback otherwise — see the two functions above."""
    llm_result = _classify_clause_llm(text)
    return llm_result if llm_result is not None else _classify_clause_keyword(text)
