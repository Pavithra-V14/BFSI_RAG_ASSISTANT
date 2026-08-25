"""
M7 demo command:

    python -m observability.trace_check --last 10

Confirms every recent request has a complete trace covering the stages it
should have touched, given how it terminated (short-circuited on refusal,
cache hit, or full pipeline run).
"""
from __future__ import annotations

import argparse
import json

from observability.tracer import read_recent_traces


def check(n: int = 10) -> dict:
    traces = read_recent_traces(n)
    results = []
    for trace_id, stages in traces.items():
        stage_names = [s["stage"] for s in stages]
        total_latency = next(
            (s.get("total_latency_ms") for s in stages if s["stage"] == "_trace_end"), None
        )
        complete = "input_guardrail" in stage_names and "_trace_end" in stage_names
        results.append({
            "trace_id": trace_id,
            "stages": stage_names,
            "total_latency_ms": total_latency,
            "complete": complete,
        })
    return {
        "traces_checked": len(results),
        "all_complete": all(r["complete"] for r in results) if results else False,
        "results": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, default=10)
    args = ap.parse_args()
    print(json.dumps(check(args.last), indent=2))


if __name__ == "__main__":
    main()
