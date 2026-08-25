"""
Incremental reindexing (M2). On re-ingest of an amended document, only
chunks whose content_hash changed get re-embedded; unchanged chunks are
left alone; removed chunks get `effective_to` set instead of being deleted
(audit trail requirement — see invariant in context-graph.json).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from ingestion.chunker import Chunk


@dataclass
class DiffResult:
    new: list[Chunk]
    changed: list[Chunk]
    unchanged: list[Chunk]
    removed_chunk_ids: list[str]


def diff_chunks(old_chunks: list[Chunk], new_chunks: list[Chunk]) -> DiffResult:
    """
    Compares by chunk_id first (position-stable clause numbering), falling
    back to content_hash to detect a same-id-different-content edit.
    """
    old_by_id = {c.chunk_id: c for c in old_chunks}
    new_by_id = {c.chunk_id: c for c in new_chunks}

    new_list, changed_list, unchanged_list = [], [], []
    for chunk_id, new_chunk in new_by_id.items():
        old_chunk = old_by_id.get(chunk_id)
        if old_chunk is None:
            new_list.append(new_chunk)
        elif old_chunk.content_hash != new_chunk.content_hash:
            new_chunk.version = old_chunk.version + 1
            changed_list.append(new_chunk)
        else:
            unchanged_list.append(old_chunk)

    removed_ids = [cid for cid in old_by_id if cid not in new_by_id]

    return DiffResult(
        new=new_list,
        changed=changed_list,
        unchanged=unchanged_list,
        removed_chunk_ids=removed_ids,
    )


def apply_diff(store, diff: DiffResult) -> None:
    """Writes only the changed surface to the store — this IS the cost win."""
    for chunk in diff.new + diff.changed:
        store.upsert_chunk(chunk)
    today = date.today().isoformat()
    for chunk_id in diff.removed_chunk_ids:
        store.set_effective_to(chunk_id, today)
