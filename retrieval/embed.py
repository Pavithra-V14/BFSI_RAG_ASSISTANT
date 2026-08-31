from __future__ import annotations

import hashlib
import logging
import os
import re
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

DIM = 768 
_TOKEN_RE = re.compile(r"[a-z0-9]+")

def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())

def _hash_embed(text: str) -> np.ndarray:
    """Deterministic hashed bag-of-words embedding. See module docstring."""
    vec = np.zeros(DIM, dtype=np.float32)
    for tok in _tokenize(text):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        idx = h % DIM
        sign = 1.0 if (h // DIM) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec

@lru_cache(maxsize=1)
def _get_gemini_client():
    if not os.environ.get("GEMINI_API_KEY"):
        return None
    try:
        from google import genai
        return genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    except Exception as e:  
        logger.warning("Gemini client init failed (%s) — embeddings falling back to hash", str(e)[:200])
        return None

def _gemini_embed(text: str) -> np.ndarray:
    from google.genai import types
    client = _get_gemini_client()
    if client is None:
        raise RuntimeError("Gemini client unavailable")
    response = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=DIM),
    )
    vec = np.array(response.embeddings[0].values, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

def embed_text(text: str) -> np.ndarray:
    """
    Real Gemini embedding when GEMINI_API_KEY is configured, hashed
    bag-of-words fallback otherwise or on any per-call failure (rate
    limit, transient network error, etc.) — a single failed embedding
    call must not crash a 25-chunk ingestion run partway through.
    """
    vec, _source = embed_text_with_source(text)
    return vec

def embed_text_with_source(text: str) -> tuple[np.ndarray, str]:
    """
    Same embedding logic as embed_text(), but also returns which backend
    actually produced the vector ("gemini" or "hash_fallback"). Ingestion
    uses this to flag, per chunk, when a degraded embedding was used
    instead of silently producing a corpus where some chunks are
    semantically searchable and others are only keyword-searchable with
    no visible difference to whoever ran the ingest.
    """
    if os.environ.get("GEMINI_API_KEY"):
        try:
            return _gemini_embed(text), "gemini"
        except Exception as e:  
            logger.warning("Gemini embed_content failed (%s), using hash fallback for this call", str(e)[:200])
    return _hash_embed(text), "hash_fallback"

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)
