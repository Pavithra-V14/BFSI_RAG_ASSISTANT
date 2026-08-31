from __future__ import annotations

from dataclasses import dataclass
from venv import logger

import config

from rank_bm25 import BM25Okapi

from ingestion.store import DocStore, VectorStore
from retrieval.embed import embed_text

RRF_K = 60  

@dataclass
class Candidate:
    chunk_id: str
    score: float
    metadata: dict
    text: str

def _bm25_rank(query: str, chunk_records: list[tuple[str, dict, str]]) -> dict[str, int]:
    """
    Returns chunk_id -> rank (0 = best) using BM25 over the pre-filtered set.

    Guards against TWO distinct empty-input cases, not just one:
      1. chunk_records itself is empty (no dense hits at all) — the
         original guard.
      2. chunk_records is non-empty but every chunk's TEXT is empty —
         happens when Qdrant returns a chunk_id that DocStore has no
         record for (the two stores have drifted out of sync — e.g. a
         fresh local DocStore JSON file pointed at a persistent remote
         Qdrant collection that already has older data in it). BM25Okapi
         itself divides by zero in this case (zero total terms across
         the whole corpus), which is the crash this second guard prevents.
    """
    if not chunk_records:
        return {}
    corpus = [text.lower().split() for _, _, text in chunk_records]
    if not any(corpus):  
        logger.warning(
            "BM25 corpus has %d chunk_ids but zero non-empty texts — "
            "DocStore/VectorStore are likely out of sync for these chunks. "
            "Skipping the sparse (BM25) pass for this query; dense retrieval still applies.",
            len(chunk_records),
        )
        return {}
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query.lower().split())
    order = sorted(range(len(chunk_records)), key=lambda i: scores[i], reverse=True)
    return {chunk_records[i][0]: rank for rank, i in enumerate(order)}


def hybrid_retrieve(
    query: str,
    vector_store: VectorStore,
    doc_store: DocStore,
    filters: dict | None = None,
    top_k: int = config.RETRIEVAL_TOP_K,
) -> list[Candidate]:
    query_vec = embed_text(query)

    dense_hits = vector_store.query(query_vec, filters=filters, top_k=max(top_k, 50))
    if not dense_hits:
        return []

    dense_rank = {chunk_id: rank for rank, (chunk_id, _, _) in enumerate(dense_hits)}

    chunk_records = []
    for chunk_id, _, meta in dense_hits:
        doc = doc_store.get(chunk_id)
        text = doc["text"] if doc else ""
        chunk_records.append((chunk_id, meta, text))
    sparse_rank = _bm25_rank(query, chunk_records)

    fused_scores: dict[str, float] = {}
    for chunk_id in dense_rank:
        d = dense_rank.get(chunk_id)
        s = sparse_rank.get(chunk_id)
        score = 0.0
        if d is not None:
            score += 1.0 / (RRF_K + d)
        if s is not None:
            score += 1.0 / (RRF_K + s)
        fused_scores[chunk_id] = score

    fused_order = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    meta_by_id = {chunk_id: meta for chunk_id, _, meta in dense_hits}
    text_by_id = {chunk_id: text for chunk_id, meta, text in chunk_records}

    return [
        Candidate(chunk_id=cid, score=score, metadata=meta_by_id[cid], text=text_by_id.get(cid, ""))
        for cid, score in fused_order
    ]
