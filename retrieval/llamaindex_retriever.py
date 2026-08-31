from __future__ import annotations

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

import config
from ingestion.store import DocStore, VectorStore
from retrieval.index import hybrid_retrieve
from retrieval.rerank import rerank

class BFSIHybridRetriever(BaseRetriever):
    """LlamaIndex-native retriever backed by our Qdrant hybrid pipeline."""

    def __init__(
        self,
        vector_store: VectorStore,
        doc_store: DocStore,
        filters: dict | None = None,
        top_k: int = config.RETRIEVAL_TOP_K,
        rerank_top_k: int = config.RERANK_TOP_K,
    ):
        self._vector_store = vector_store
        self._doc_store = doc_store
        self._filters = filters or {}
        self._top_k = top_k
        self._rerank_top_k = rerank_top_k
        self.last_candidate_count = 0  
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        candidates = hybrid_retrieve(
            query_bundle.query_str, self._vector_store, self._doc_store,
            filters=self._filters, top_k=self._top_k,
        )
        self.last_candidate_count = len(candidates)
        reranked = rerank(query_bundle.query_str, candidates, top_k=self._rerank_top_k)
        return [
            NodeWithScore(
                node=TextNode(text=c.text, id_=c.chunk_id, metadata=c.metadata),
                score=c.score,
            )
            for c in reranked
        ]
