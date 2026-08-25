"""
M8 demo command:

    python -m cache.smoke_test --role-a claims_adjuster --role-b relationship_manager

Tests the CORRECT invariant after the 2026-08-23 scoping fix: cache
entries are shared across callers with the SAME role, and never leak
across DIFFERENT roles — this is what actually matters, since role is
the dimension that determines document access (see retrieval/cache.py's
module docstring for the bug this replaced: a query-content heuristic
that let one role's cached answer leak to a different, differently-
privileged role).
"""
from __future__ import annotations

import argparse
import json

from retrieval.cache import SemanticCache


def run(role_a: str, role_b: str) -> dict:
    cache = SemanticCache()

    # cached under role_a
    cache.store(
        query="what is the co-payment for out-of-network treatment",
        role=role_a,
        answer="A co-payment of 10% applies outside the network.",
        citations=["policy_wording — sample_health_policy::c025"],
        chunk_versions={"sample_health_policy::c025": 1},
    )

    same_role_hit = cache.lookup("what is the co-payment for out-of-network treatment", role=role_a)
    cross_role_hit = cache.lookup("what is the co-payment for out-of-network treatment", role=role_b)

    removed = cache.invalidate_for_chunk("sample_health_policy::c025", new_version=2)
    after_invalidate = cache.lookup("what is the co-payment for out-of-network treatment", role=role_a)

    return {
        "same_role_cache_hit": same_role_hit is not None,       # must be True
        "cross_role_leak_detected": cross_role_hit is not None,  # must be False
        "entries_invalidated_on_reingest": removed,
        "entry_gone_after_invalidate": after_invalidate is None,
        "PASS": (
            same_role_hit is not None
            and cross_role_hit is None
            and removed >= 1
            and after_invalidate is None
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role-a", default="claims_adjuster")
    ap.add_argument("--role-b", default="relationship_manager")
    args = ap.parse_args()
    print(json.dumps(run(args.role_a, args.role_b), indent=2))


if __name__ == "__main__":
    main()
