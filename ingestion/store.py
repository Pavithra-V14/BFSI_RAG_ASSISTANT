from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ingestion.chunker import Chunk
from retrieval.embed import DIM, embed_text

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

QDRANT_PATH = DATA_DIR / "qdrant_local"
DOC_STORE_PATH = DATA_DIR / "doc_store.json"

COLLECTION = "bfsi_chunks"
FILTERABLE_FIELDS = [
    "chunk_id", "doc_id", "source", "effective_date",
    "product_line", "clause_type", "version", "access_role", "effective_to",
]


def _stable_point_id(chunk_id: str) -> int:
    """Qdrant point IDs must be int or UUID — derive a stable int from chunk_id."""
    import hashlib
    return int(hashlib.sha256(chunk_id.encode()).hexdigest()[:12], 16)

def _build_client(path: Path) -> QdrantClient:
    url = os.environ.get("QDRANT_URL")
    if url:
        api_key = os.environ.get("QDRANT_API_KEY") or None
        return QdrantClient(url=url, api_key=api_key)
    return QdrantClient(path=str(path))

class VectorStore:
    def __init__(self, path: Path = QDRANT_PATH):
        self.client = _build_client(path)
        self._ensure_collection()
        self._ensure_payload_indexes()

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if COLLECTION not in existing:
            self.client.create_collection(
                collection_name=COLLECTION,
                vectors_config=qmodels.VectorParams(size=DIM, distance=qmodels.Distance.COSINE),
            )

    def _ensure_payload_indexes(self) -> None:
        """
        Remote Qdrant (Qdrant Cloud or any real server) requires an
        explicit payload index before you can filter or scroll on a field
        — embedded/local mode is more permissive and doesn't enforce this,
        which is why role-filtering and re-ingestion worked fine locally
        and only broke after switching to Qdrant Cloud (error: "Index
        required but not found for ... Help: Create an index for this
        key"). Creating indexes is idempotent-safe here: if one already
        exists, Qdrant returns an error we simply ignore.
        """
        keyword_fields = ["doc_id", "source", "product_line", "clause_type", "effective_to"]
        for field in keyword_fields:
            try:
                self.client.create_payload_index(
                    collection_name=COLLECTION,
                    field_name=field,
                    field_schema=qmodels.PayloadSchemaType.KEYWORD,
                )
            except Exception: 
                pass
            
        try:
            self.client.create_payload_index(
                collection_name=COLLECTION,
                field_name="access_role",
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
        except Exception:  
            pass

    def upsert_chunk(self, chunk: Chunk, embedding: np.ndarray | None = None) -> None:
        vec = embedding if embedding is not None else embed_text(chunk.text)
        payload = {f: getattr(chunk, f) for f in FILTERABLE_FIELDS}
        self.client.upsert(
            collection_name=COLLECTION,
            points=[qmodels.PointStruct(
                id=_stable_point_id(chunk.chunk_id),
                vector=vec.tolist(),
                payload=payload,
            )],
        )

    def set_effective_to(self, chunk_id: str, date_str: str) -> None:
        self.client.set_payload(
            collection_name=COLLECTION,
            payload={"effective_to": date_str},
            points=[_stable_point_id(chunk_id)],
        )

    def query(
        self,
        query_embedding: np.ndarray,
        filters: dict | None = None,
        top_k: int = 25,
        include_superseded: bool = False,
    ) -> list[tuple[str, float, dict]]:
        """Pre-filter THEN score — never post-filter (see invariant)."""
        qfilter = self._build_filter(filters or {}, include_superseded)
        hits = self.client.query_points(
            collection_name=COLLECTION,
            query=query_embedding.tolist(),
            query_filter=qfilter,
            limit=top_k,
        ).points
        return [(h.payload["chunk_id"], h.score, h.payload) for h in hits]

    @staticmethod
    def _build_filter(filters: dict, include_superseded: bool) -> qmodels.Filter | None:
        must = []
        for key, allowed in filters.items():
            if allowed is None:
                continue
            allowed_list = allowed if isinstance(allowed, list) else [allowed]
            must.append(qmodels.FieldCondition(key=key, match=qmodels.MatchAny(any=allowed_list)))
        if not include_superseded:
            must.append(qmodels.IsNullCondition(is_null=qmodels.PayloadField(key="effective_to")))
        if not must:
            return None
        return qmodels.Filter(must=must)

    def all_chunk_ids_for_doc(self, doc_id: str) -> list[str]:
        records, _ = self.client.scroll(
            collection_name=COLLECTION,
            scroll_filter=qmodels.Filter(
                must=[qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc_id))]
            ),
            limit=1000,
            with_payload=True,
        )
        return [r.payload["chunk_id"] for r in records]

    def get_by_chunk_id(self, chunk_id: str) -> dict | None:
        pts = self.client.retrieve(collection_name=COLLECTION, ids=[_stable_point_id(chunk_id)])
        return pts[0].payload if pts else None


class DocStore:
    """Kept as local JSON — production swap-point is Postgres/DynamoDB, per
    ADR 0002 / ARCHITECTURE §1.4. Full chunk text never goes into Qdrant."""

    def __init__(self, path: Path = DOC_STORE_PATH):
        self.path = path
        self._records: dict[str, dict] = self._load()

    def _load(self) -> dict[str, dict]:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {}

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._records, indent=2))

    def upsert_chunk(self, chunk: Chunk) -> None:
        self._records[chunk.chunk_id] = asdict(chunk)
        self._save()

    def get(self, chunk_id: str) -> dict | None:
        return self._records.get(chunk_id)

    def get_many(self, chunk_ids: list[str]) -> list[dict]:
        return [self._records[cid] for cid in chunk_ids if cid in self._records]


def get_stores() -> tuple[VectorStore, DocStore]:
    return VectorStore(), DocStore()
