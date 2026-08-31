from __future__ import annotations

import logging
import os
import re

import config
from retrieval.index import Candidate

logger = logging.getLogger(__name__)

RERANK_TOP_K = config.RERANK_TOP_K  
RELEVANCE_FLOOR = config.RELEVANCE_FLOOR

_LOCAL_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
_JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
_JINA_DEFAULT_MODEL = "jina-reranker-v2-base-multilingual"  

def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))

def _lexical_rerank(query: str, candidates: list[Candidate], top_k: int) -> list[Candidate]:
    """Fallback reranker — no API key, no model download required. See
    module docstring."""
    q_tokens = _tokens(query)
    if not q_tokens or not candidates:
        return candidates[:top_k]
    scored = []
    for c in candidates:
        c_tokens = _tokens(c.text)
        overlap = len(q_tokens & c_tokens) / max(len(q_tokens), 1)
        boost = 0.1 if c.metadata.get("clause_type", "") in query.lower() else 0.0
        scored.append((overlap + boost, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [_rescored(c, s) for s, c in scored[:top_k]]

def _rescored(candidate: Candidate, new_score: float) -> Candidate:
    return Candidate(
        chunk_id=candidate.chunk_id, score=new_score,
        metadata=candidate.metadata, text=candidate.text,
    )

def _cohere_rerank(query: str, candidates: list[Candidate], top_k: int) -> list[Candidate]:
    import cohere
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("COHERE_API_KEY not set")

    client = cohere.ClientV2(api_key=api_key)
    docs = [c.text for c in candidates]
    response = client.rerank(model="rerank-v3.5", query=query, documents=docs, top_n=top_k)
    return [
        _rescored(candidates[r.index], r.relevance_score)
        for r in response.results
    ]

def _jina_rerank(query: str, candidates: list[Candidate], top_k: int) -> list[Candidate]:
    import requests

    api_key = os.environ.get("JINA_API_KEY")
    if not api_key:
        raise RuntimeError("JINA_API_KEY not set")

    model = os.environ.get("JINA_RERANK_MODEL", _JINA_DEFAULT_MODEL)
    docs = [c.text for c in candidates]
    response = requests.post(
        _JINA_RERANK_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "query": query, "documents": docs, "top_n": top_k},
        timeout=10,
    )
    response.raise_for_status()
    results = response.json()["results"]
    return [
        _rescored(candidates[r["index"]], r["relevance_score"])
        for r in results
    ]

def _local_reranker_enabled() -> bool:
    return os.environ.get("LOCAL_RERANKER_ENABLED", "false").lower() in ("true", "1", "yes")

def _get_local_reranker():
    """
    Lazily loads bge-reranker-v2-m3 once per process (module-level cache
    via the closure below), same reachability-first pattern as LLM
    Guard's scanner loader — fails fast in milliseconds if the model hub
    is unreachable rather than hanging on a full connection timeout.
    """
    if not hasattr(_get_local_reranker, "_cached"):
        import socket
        try:
            socket.setdefaulttimeout(1.5)
            socket.gethostbyname("huggingface.co")
        except OSError:
            logger.warning(
                "huggingface.co unreachable — local reranker unavailable, "
                "falling through to lexical reranker."
            )
            _get_local_reranker._cached = None
            return None

        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading local reranker %s (first use only, cached after)", _LOCAL_RERANKER_MODEL)
            _get_local_reranker._cached = CrossEncoder(_LOCAL_RERANKER_MODEL)
        except Exception as e:  
            logger.warning("Local reranker load failed (%s), falling through to lexical reranker", str(e)[:200])
            _get_local_reranker._cached = None

    return _get_local_reranker._cached


def _local_rerank(query: str, candidates: list[Candidate], top_k: int) -> list[Candidate]:
    model = _get_local_reranker()
    if model is None:
        raise RuntimeError("local reranker unavailable")
    pairs = [(query, c.text) for c in candidates]
    scores = model.predict(pairs)
    scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [_rescored(c, float(s)) for s, c in scored[:top_k]]


def rerank(query: str, candidates: list[Candidate], top_k: int = config.RERANK_TOP_K) -> list[Candidate]:
    if not candidates:
        return []

    if os.environ.get("COHERE_API_KEY"):
        try:
            return _cohere_rerank(query, candidates, top_k)
        except Exception as e:  
            logger.warning("Cohere rerank failed (%s), trying next tier", str(e)[:300])

    if os.environ.get("JINA_API_KEY"):
        try:
            return _jina_rerank(query, candidates, top_k)
        except Exception as e: 
            logger.warning("Jina rerank failed (%s), trying next tier", str(e)[:300])

    if _local_reranker_enabled():
        try:
            return _local_rerank(query, candidates, top_k)
        except Exception as e: 
            logger.warning("Local reranker failed (%s), falling back to lexical reranker", str(e)[:200])

    if not any([os.environ.get("COHERE_API_KEY"), os.environ.get("JINA_API_KEY"), _local_reranker_enabled()]):
        logger.info(
            "No reranker configured (COHERE_API_KEY, JINA_API_KEY, LOCAL_RERANKER_ENABLED all unset) "
            "— using lexical fallback"
        )
    return _lexical_rerank(query, candidates, top_k)