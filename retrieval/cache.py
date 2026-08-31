from __future__ import annotations

import json
import logging
import os
import time

import numpy as np
import valkey

import config
from retrieval.embed import cosine_similarity, embed_text

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = config.CACHE_SIMILARITY_THRESHOLD
DEFAULT_TTL_SECONDS = config.CACHE_DEFAULT_TTL_SECONDS

def _client() -> valkey.Valkey:
    uri = os.environ.get("VALKEY_URI")
    if uri:
        ca_cert = os.environ.get("VALKEY_CA_CERT")
        kwargs = {"decode_responses": True}
        if ca_cert:
            kwargs["ssl_ca_certs"] = ca_cert
            kwargs["ssl_cert_reqs"] = "required"
        elif uri.startswith("rediss://"):
            kwargs["ssl_cert_reqs"] = "none"
        return valkey.Valkey.from_url(uri, **kwargs)

    use_ssl = os.environ.get("VALKEY_SSL", "false").lower() == "true"
    ca_cert = os.environ.get("VALKEY_CA_CERT")
    kwargs = {
        "host": os.environ.get("VALKEY_HOST", "localhost"),
        "port": int(os.environ.get("VALKEY_PORT", "6379")),
        "decode_responses": True,
    }
    password = os.environ.get("VALKEY_PASSWORD")
    if password:
        kwargs["password"] = password
    if use_ssl:
        kwargs["ssl"] = True
        if ca_cert:
            kwargs["ssl_ca_certs"] = ca_cert
            kwargs["ssl_cert_reqs"] = "required"
        else:
            kwargs["ssl_cert_reqs"] = "none"
    return valkey.Valkey(**kwargs)

class SemanticCache:
    def __init__(self):
        try:
            self._redis = _client()
            self._redis.ping()
        except Exception as e: 
            logger.warning("Valkey unreachable (%s) — cache will no-op (misses only, never errors)", str(e)[:200])
            self._redis = None

    @staticmethod
    def _key(role: str, query: str) -> str:
        query_hash = str(abs(hash(query)))
        return f"cache:role:{role}:{query_hash}"

    def _scan_scope(self, role: str) -> list[str]:
        if self._redis is None:
            return []
        return list(self._redis.scan_iter(f"cache:role:{role}:*"))

    def lookup(self, query: str, role: str):
        if self._redis is None:
            return None
        query_vec = embed_text(query)
        best, best_score = None, 0.0
        stale_keys: list[str] = []
        for key in self._scan_scope(role):
            raw = self._redis.get(key)
            if not raw:
                continue
            entry = json.loads(raw)
            try:
                score = cosine_similarity(query_vec, np.array(entry["embedding"]))
            except ValueError:
                logger.warning("Stale cache entry with mismatched embedding dims, evicting: %s", key)
                stale_keys.append(key)
                continue
            if score > best_score:
                best, best_score = entry, score
        for key in stale_keys:
            self._redis.delete(key)
        if best and best_score >= SIMILARITY_THRESHOLD:
            return best
        return None

    def store(
        self,
        query: str,
        role: str,
        answer: str,
        citations: list[str],
        chunk_versions: dict[str, int],
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        generation_provider: str | None = None,
    ) -> None:
        if self._redis is None:
            return
        key = self._key(role, query)
        entry = {
            "query": query,
            "embedding": embed_text(query).tolist(),
            "answer": answer,
            "citations": citations,
            "chunk_versions": chunk_versions,
            "created_at": time.time(),
            "generation_provider": generation_provider, 
        }
        self._redis.set(key, json.dumps(entry), ex=ttl_seconds)

    def invalidate_for_chunk(self, chunk_id: str, new_version: int) -> int:
        """Called when a document is re-ingested (M2) — drops any cache
        entry that cited a now-superseded chunk version. Returns count
        removed, for observability."""
        if self._redis is None:
            return 0
        removed = 0
        for key in list(self._redis.scan_iter("cache:*")):
            raw = self._redis.get(key)
            if not raw:
                continue
            entry = json.loads(raw)
            if entry.get("chunk_versions", {}).get(chunk_id, new_version) < new_version:
                self._redis.delete(key)
                removed += 1
        return removed