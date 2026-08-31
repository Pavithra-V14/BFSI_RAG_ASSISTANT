from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from ingestion.chunker import clause_chunk, multimodal_chunks
from ingestion.parser import parse
from ingestion.safety_scan import scan_document
from ingestion.store import DocStore, VectorStore
from ingestion.versioning import apply_diff, diff_chunks
from retrieval.embed import embed_text_with_source
from concurrent.futures import ThreadPoolExecutor, as_completed
import config

def run_ingestion(
    doc_path: str,
    source: str,
    product_line: str,
    dry_run: bool = False,
    access_role: list[str] | None = None,
    vector_store: VectorStore | None = None,
    doc_store: DocStore | None = None,
) -> dict:
    """
    vector_store/doc_store: pass the caller's already-open instances when
    one exists in-process (e.g. app.py's module-level singletons) — Qdrant's
    embedded local mode locks its storage path to a single client per
    process, so opening a second VectorStore() while one is already open
    raises. The CLI entrypoint (main(), below) has no existing instance and
    opens its own.
    """
    parsed = parse(doc_path)

    scan = scan_document(parsed.raw_bytes, parsed.text)
    if not scan.safe:
        return {
            "status": "quarantined",
            "doc_id": parsed.doc_id,
            "reasons": scan.reasons,
        }

    new_chunks = clause_chunk(
        parsed.text, doc_id=parsed.doc_id, source=source, product_line=product_line
    )

    multimodal = multimodal_chunks(
        parsed.tables_as_markdown, parsed.image_captions,
        doc_id=parsed.doc_id, source=source, product_line=product_line,
        start_seq=len(new_chunks),
    )
    new_chunks += multimodal

    if access_role:
        for c in new_chunks:
            c.access_role = access_role

    owns_stores = vector_store is None and doc_store is None
    if owns_stores:
        vector_store, doc_store = VectorStore(), DocStore()

    existing_ids = vector_store.all_chunk_ids_for_doc(parsed.doc_id)
    existing_chunks = []
    for cid in existing_ids:
        rec = doc_store.get(cid)
        if rec:
            from ingestion.chunker import Chunk
            existing_chunks.append(Chunk(**{k: rec[k] for k in Chunk.__dataclass_fields__}))

    diff = diff_chunks(existing_chunks, new_chunks)

    chunks_to_embed = diff.new + diff.changed
    embedding_sources: dict[str, str] = {}
    embedding_cache: dict[str, tuple] = {}

    if chunks_to_embed:
        with ThreadPoolExecutor(max_workers=config.INGESTION_EMBED_CONCURRENCY) as pool:
            future_to_chunk = {
                pool.submit(embed_text_with_source, chunk.text): chunk
                for chunk in chunks_to_embed
            }
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                vec, embed_source = future.result()
                embedding_sources[chunk.chunk_id] = embed_source
                embedding_cache[chunk.chunk_id] = vec

    degraded_chunk_ids = [cid for cid, src in embedding_sources.items() if src == "hash_fallback"]
    
    result = {
        "status": "quarantined" if not scan.safe else "ok",
        "doc_id": parsed.doc_id,
        "new_chunks": len(diff.new),
        "changed_chunks": len(diff.changed),
        "unchanged_chunks": len(diff.unchanged),
        "removed_chunks": len(diff.removed_chunk_ids),
        "multimodal_summary": {
            "tables_extracted": len(parsed.tables_as_markdown),
            "images_captioned": len(parsed.image_captions),
            "images_skipped_no_caption": parsed.images_skipped_no_caption,
        },
        "embedding_summary": {
            "gemini": sum(1 for s in embedding_sources.values() if s == "gemini"),
            "hash_fallback": len(degraded_chunk_ids),
        },
        "degraded_chunk_ids": degraded_chunk_ids, 
        "sample_metadata": [
            {
                "chunk_id": c.chunk_id,
                "clause_type": c.clause_type,
                "content_hash": c.content_hash,
                "source": c.source,
                "effective_date": c.effective_date,
                "product_line": c.product_line,
                "version": c.version,
                "embedding_source": embedding_sources.get(c.chunk_id),
            }
            for c in (diff.new + diff.changed)[:3]
        ],
    }

    if not dry_run:
        class _CombinedStore:
            def upsert_chunk(self, chunk):
                vector_store.upsert_chunk(chunk, embedding=embedding_cache[chunk.chunk_id])
                doc_store.upsert_chunk(chunk)

            def set_effective_to(self, chunk_id, date_str):
                vector_store.set_effective_to(chunk_id, date_str)

        apply_diff(_CombinedStore(), diff)

    return result


def deprecate_document(
    doc_id: str,
    vector_store: VectorStore | None = None,
    doc_store: DocStore | None = None,
) -> dict:
    """
    Retracts a document WITHOUT requiring a replacement version — sets
    effective_to on every one of its chunks (soft-close, same mechanism
    M2's versioning already uses for a superseded chunk), so it stops
    being retrieved but stays queryable with include_superseded=True for
    audit replay. Distinct from re-ingesting an amended document: this is
    for "this regulation was fully withdrawn," not "this regulation was
    updated."
    """
    owns_stores = vector_store is None and doc_store is None
    if owns_stores:
        vector_store, doc_store = VectorStore(), DocStore()

    chunk_ids = vector_store.all_chunk_ids_for_doc(doc_id)
    if not chunk_ids:
        return {"status": "not_found", "doc_id": doc_id, "chunks_deprecated": 0}

    today = date.today().isoformat()
    for cid in chunk_ids:
        vector_store.set_effective_to(cid, today)

    return {"status": "deprecated", "doc_id": doc_id, "chunks_deprecated": len(chunk_ids), "effective_to": today}


def get_stores_for(dry_run: bool) -> tuple[VectorStore, DocStore]:
    return VectorStore(), DocStore()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", required=True)
    ap.add_argument("--source", default="policy_wording")
    ap.add_argument("--product-line", default="health")
    ap.add_argument("--access-role", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not Path(args.doc).exists():
        raise SystemExit(f"file not found: {args.doc}")

    result = run_ingestion(
        args.doc, args.source, args.product_line, args.dry_run, args.access_role
    )
    import json
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
