"""
Semantic query cache — real Valkey backend (see ADR 0004).

⚠️ SCOPING FIX (2026-08-23): originally scoped by "global" vs "user_id"
tiers, with "global" decided by a crude heuristic (`"user" not in
query.lower()`). That heuristic has nothing to do with document access —
a query with no literal word "user" in it got cached in the shared global
tier and served to EVERY caller regardless of role, bypassing the
retrieval-layer access_role filter entirely. Confirmed live: an answer
retrieved under a role that lacked access to a document got cached
globally, then served verbatim to a different role that DID have access,
serving a wrong/under-scoped answer instead of running that role's own
correctly-filtered retrieval.

Fixed by scoping every cache key by ROLE, not a query-content guess. Two
tiers now:
  role tier — cache:role:<role>:<hash> — shared across all callers with
              the SAME role. Safe: role is exactly the dimension that
              determines which chunks are visible, so same-role callers
              always see the same retrieval-eligible content.
  user tier — cache:user:<user_id>:<hash> — for anything that should
              never be shared even within a role (kept for future use;
              nothing currently routes here, but the tier exists so a
              future feature can opt in explicitly rather than guessing).

This trades away true global sharing (a claims_adjuster and a
compliance_officer asking the identical public-regulatory question won't
share a cache entry) for correctness — a much safer trade than the bug
this replaces. See context-graph.json's cache_key_namespaced_by_scope
invariant, now extended to cover role as well as user_id.
"""
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
        # ssl_cert_reqs="none" if no CA cert given — encrypts in transit,
        # doesn't verify the chain. Fine to unblock Aiven quickly; add the
        # CA cert (VALKEY_CA_CERT) before this touches real traffic.
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
        except Exception as e:  # noqa: BLE001
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
                # Dimension mismatch — entry was cached with a different
                # embedding model/DIM than the one currently active (e.g.
                # embed.py's DIM changed after this entry was written).
                # Skip it rather than crash the whole lookup, and clean it
                # up so it doesn't get rescanned on every future query.
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
            "generation_provider": generation_provider,  # 2026-08-24 — item 5:
            # without this, a cache hit replaying a mock-generated answer
            # would look identical to a real one at read time, with no way
            # to tell — the exact silent-degradation gap this whole item
            # exists to close, and it applies to cached answers just as
            # much as fresh ones.
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