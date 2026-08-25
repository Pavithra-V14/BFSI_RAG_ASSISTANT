"""
Retrieval query CLI — M3 demo command.

    python -m retrieval.query --q "portability waiting period" \
        --role claims_adjuster --explain
"""
from __future__ import annotations

import argparse
import json

from guardrails.context_guardrail import check_retrieved_context
from guardrails.input_guardrail import check_input
from ingestion.store import DocStore, VectorStore
from retrieval.index import hybrid_retrieve
from retrieval.rerank import rerank
import config


def run_query(query: str, role: str, product_line: str | None = None, explain: bool = False) -> dict:
    vector_store, doc_store = VectorStore(), DocStore()

    # First pass at low top_k just to estimate confidence for the guardrail
    probe = hybrid_retrieve(query, vector_store, doc_store, filters={"access_role": [role, "*"]}, top_k=5)
    probe_confidence = probe[0].score if probe else 0.0

    decision = check_input(query, retrieval_confidence=probe_confidence)
    if decision.action in ("refuse", "block"):
        return {"decision": decision.action, "reason": decision.reason, "results": []}

    effective_query = decision.query

    filters = {"access_role": [role, "*"]}
    if product_line:
        filters["product_line"] = product_line

    candidates = hybrid_retrieve(effective_query, vector_store, doc_store, filters=filters, top_k=config.RETRIEVAL_TOP_K)
    reranked = rerank(effective_query, candidates)

    context_check = check_retrieved_context(reranked, user_role=role)
    if not context_check.passed:
        return {
            "decision": "fail_closed",
            "reason": context_check.reason,
            "dropped": context_check.dropped,
            "results": [],
        }
    reranked = context_check.surviving_chunks

    out = {
        "decision": decision.action,
        "effective_query": effective_query,
        "candidate_count": len(candidates),
        "context_guardrail_dropped": context_check.dropped,
        "final_chunks": [
            {
                "chunk_id": c.chunk_id,
                "score": round(c.score, 4),
                "clause_type": c.metadata.get("clause_type"),
                "source": c.metadata.get("source"),
                "text_preview": c.text[:140],
            }
            for c in reranked
        ],
    }
    if explain:
        out["candidates_before_rerank"] = [
            {"chunk_id": c.chunk_id, "rrf_score": round(c.score, 4)} for c in candidates
        ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True)
    ap.add_argument("--role", default="*")
    ap.add_argument("--product-line", default=None)
    ap.add_argument("--explain", action="store_true")
    args = ap.parse_args()

    result = run_query(args.q, args.role, args.product_line, args.explain)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
