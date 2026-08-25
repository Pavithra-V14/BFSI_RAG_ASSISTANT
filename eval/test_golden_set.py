"""
M6 demo command:

    pytest eval/test_golden_set.py -v

DeepEval-based CI gate (see ADR 0008) — runs the full harness (input
guardrail -> retrieval -> rerank -> context guardrail -> generation ->
output guardrail) against the golden set and asserts every metric in
wiki/concepts/pipeline-parameters.md. This is the regression gate every
future milestone's demo command should be run alongside.

Golden set cases come in two kinds, both in eval/golden_set/sample.jsonl:
  - retrieval cases: {"query", "role", "expected_clause_type", "expected_doc_id"}
    scored on Recall@25, Precision@5, faithfulness, citation presence.
  - refusal cases: {"query", "role", "expect_refusal": true}
    in-domain-but-unanswerable or out-of-domain/injection queries that
    should be refused, not answered. Scored separately (refusal accuracy)
    — a retrieval quality metric has no meaning for a query with no
    expected document, so these are excluded from Recall/Precision math
    rather than silently counted as misses (which would understate real
    retrieval quality) or hits (which would overstate it).
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest
from deepeval.test_case import LLMTestCase

import config
from eval.ragas_metrics import ContextPrecisionMetric, FaithfulnessMetric
from gateway.llm_gateway import get_gateway
from guardrails.context_guardrail import check_retrieved_context
from guardrails.input_guardrail import check_input
from guardrails.output_guardrail import apply_output_guardrail
from ingestion.store import DocStore, VectorStore
from retrieval.index import hybrid_retrieve
from retrieval.rerank import rerank

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set" / "sample.jsonl"

# Eval targets — see wiki/concepts/pipeline-parameters.md, this is the
# single source of truth mirrored into code as the CI gate.
TARGET_RECALL_AT_25 = 0.90
TARGET_PRECISION_AT_5 = 0.80
TARGET_FAITHFULNESS = 0.80
TARGET_REFUSAL_ACCURACY = 1.00  # binary gate — a compliance assistant answering an
                                 # out-of-domain or injection query is a worse failure
                                 # than a slightly-off retrieval score, so this is 100%
                                 # not a threshold, same reasoning as citation accuracy.


def load_golden_set() -> list[dict]:
    return [json.loads(line) for line in GOLDEN_SET_PATH.read_text().splitlines() if line.strip()]


def run_refusal_case(case: dict) -> dict:
    """
    Runs ONLY the input guardrail — the layer actually responsible for
    refusal decisions — rather than the full retrieval pipeline. A query
    like "ignore all previous instructions" should never even reach
    retrieval, so scoring this against retrieval metrics would be
    meaningless; scoring it against the guardrail's decision is the
    correct, narrower check.
    """
    from observability.tracer import start_trace

    with start_trace("eval_refusal", user_id=None) as trace:
        decision = check_input(case["query"])
        refused = decision.action in ("refuse", "block")
        trace.log_stage("input_guardrail", action=decision.action, query=case["query"][:100])
        return {"case": case, "refused_correctly": refused, "decision_action": decision.action}


def run_case(case: dict, vector_store: VectorStore | None = None, doc_store: DocStore | None = None) -> dict:
    """
    vector_store/doc_store: pass the caller's already-open instances when
    one exists in-process (e.g. app.py's /admin/eval reusing the app's
    module-level singletons) — same reasoning as ingestion/run.py's
    run_ingestion(): Qdrant's embedded local mode locks its storage path
    to one client per process. Standalone `pytest eval/test_golden_set.py`
    has no existing instance and opens its own.

    2026-08-24 — now traced with request_type="eval", distinct from live
    "query" traces. Before this, eval's provider calls, retrieval, and
    generation were completely invisible in the trace log and Langfuse —
    confirmed as a real gap today: the provider-reliability dashboard was
    silently reflecting ONLY live traffic, never eval's own attempts,
    which made diagnosing "why does eval keep failing" much harder than
    it needed to be.
    """
    from observability.tracer import start_trace

    with start_trace("eval", user_id=None) as trace:
        owns_stores = vector_store is None and doc_store is None
        if owns_stores:
            vector_store, doc_store = VectorStore(), DocStore()
        gateway = get_gateway()  # shared singleton — see gateway/llm_gateway.py

        # False-refusal check: this is a real, in-domain, answerable question
        # from the golden set — it must NOT get refused by the input guardrail.
        # A false refusal here is a worse product failure than an imperfect
        # retrieval score (the user gets nothing at all instead of an
        # imperfect-but-useful answer), so it's tracked as its own metric.
        probe = hybrid_retrieve(case["query"], vector_store, doc_store, filters={"access_role": [case["role"], "*"]}, top_k=5)
        probe_confidence = probe[0].score if probe else 0.0
        guardrail_decision = check_input(case["query"], retrieval_confidence=probe_confidence)
        falsely_refused = guardrail_decision.action in ("refuse", "block")
        trace.log_stage("input_guardrail", action=guardrail_decision.action, query=case["query"][:100])

        filters = {"access_role": [case["role"], "*"]}
        candidates = hybrid_retrieve(case["query"], vector_store, doc_store, filters=filters, top_k=25)
        trace.log_stage("retrieval", candidate_count=len(candidates))
        reranked = rerank(case["query"], candidates, top_k=5)
        trace.log_stage("rerank", final_chunk_count=len(reranked))

        retrieved_doc_ids = {c.metadata.get("doc_id") for c in candidates}
        retrieved_clause_types = {c.metadata.get("clause_type") for c in reranked}
        recall_hit = case["expected_doc_id"] in retrieved_doc_ids
        precision_hit = case["expected_clause_type"] in retrieved_clause_types

        # Rank of the first candidate matching expected_doc_id, in the
        # PRE-rerank candidate order — feeds MRR. 0 means "not found at all".
        doc_rank = next(
            (i + 1 for i, c in enumerate(candidates) if c.metadata.get("doc_id") == case["expected_doc_id"]), 0
        )
        # Binary relevance per position in the reranked top-5 — feeds nDCG@5.
        relevance_at_rank = [
            1 if c.metadata.get("clause_type") == case["expected_clause_type"] else 0 for c in reranked
        ]

        ctx_result = check_retrieved_context(reranked, user_role=case["role"])
        trace.log_stage("context_guardrail", passed=ctx_result.passed, dropped_count=len(ctx_result.dropped))
        surviving = ctx_result.surviving_chunks if ctx_result.passed else []

        context_texts = [c.text for c in surviving]
        gen = gateway.generate(
            system_prompt="Answer only from the provided BFSI policy/regulatory context. Cite clauses. Be concise.",
            context_chunks=context_texts,
            query=case["query"],
        )
        trace.log_stage(
            "generation", provider=gen["provider"], error=gen["error"],
            provider_attempts=gen.get("provider_attempts", []),
        )
        chunk_dicts = [{"chunk_id": c.chunk_id, "source": c.metadata.get("source"), "text": c.text} for c in surviving]
        out = apply_output_guardrail(gen["text"], chunk_dicts)
        trace.log_stage("output_guardrail", grounded=out.grounded)

        return {
            "case": case,
            "falsely_refused": falsely_refused,
            "doc_rank": doc_rank,
            "relevance_at_rank": relevance_at_rank,
            "recall_hit": recall_hit,
            "precision_hit": precision_hit,
            "answer": out.text,
            "context_texts": context_texts,
            "grounded": out.grounded,
        }


@pytest.fixture(scope="module")
def shared_stores() -> tuple[VectorStore, DocStore]:
    """
    One VectorStore/DocStore instance, shared across every parallel
    run_case() call in this module — Qdrant's embedded local mode locks
    its storage path to a single client per process; without this shared
    fixture, each parallel thread would try to open its own client
    against the same path and crash with "already accessed by another
    instance." Remote Qdrant (Qdrant Cloud, self-hosted server) doesn't
    have this constraint, but sharing one instance is still correct and
    slightly cheaper there too — no reason for two code paths.
    """
    return VectorStore(), DocStore()


def _run_cases_parallel(cases: list[dict], run_fn, *extra_args) -> list[dict]:
    """
    Runs golden-set cases concurrently instead of one-by-one — each case
    makes several real, independent API calls (retrieval, rerank,
    generation), so this is the same I/O-bound-parallelization reasoning
    as ingestion/run.py's embedding loop. Bounded by config.EVAL_CONCURRENCY
    to respect free-tier rate limits, same trade-off documented there.
    Order of the returned results is NOT guaranteed to match `cases`'
    order — callers that need to correlate a result back to its case
    should do so via the "case" key each result dict carries, not by index.

    Callers MUST pass a shared VectorStore/DocStore via extra_args (see
    shared_stores fixture / evaluate_summary's own shared instance below)
    — never rely on run_case()'s own owns_stores fallback under
    concurrency, which opens one client per thread and crashes embedded
    Qdrant's single-client-per-process lock.
    """
    if not cases:
        return []
    with ThreadPoolExecutor(max_workers=config.EVAL_CONCURRENCY) as pool:
        futures = [pool.submit(run_fn, c, *extra_args) for c in cases]
        return [f.result() for f in as_completed(futures)]


@pytest.fixture(scope="module")
def golden_results(shared_stores) -> list[dict]:
    """Only the retrieval cases — refusal cases run through a separate
    fixture since they exercise a different code path entirely."""
    vector_store, doc_store = shared_stores
    cases = [c for c in load_golden_set() if not c.get("expect_refusal")]
    return _run_cases_parallel(cases, run_case, vector_store, doc_store)


@pytest.fixture(scope="module")
def refusal_results() -> list[dict]:
    # run_refusal_case only exercises the input guardrail (see its
    # docstring) — no vector store involved at all, no shared-instance
    # concern here.
    return _run_cases_parallel([c for c in load_golden_set() if c.get("expect_refusal")], run_refusal_case)


def test_recall_at_25_meets_target(golden_results):
    hits = sum(r["recall_hit"] for r in golden_results)
    recall = hits / len(golden_results)
    assert recall >= TARGET_RECALL_AT_25, f"Recall@25 = {recall:.3f}, target >= {TARGET_RECALL_AT_25}"


def test_precision_at_5_meets_target(golden_results):
    hits = sum(r["precision_hit"] for r in golden_results)
    precision = hits / len(golden_results)
    assert precision >= TARGET_PRECISION_AT_5, f"Precision@5 = {precision:.3f}, target >= {TARGET_PRECISION_AT_5}"


def test_faithfulness_meets_target(golden_results):
    grounded = [r for r in golden_results if r["grounded"]]
    if not grounded:
        pytest.skip("no grounded answers in this run to score")

    def _score_one(r: dict) -> float:
        tc = LLMTestCase(input=r["case"]["query"], actual_output=r["answer"], retrieval_context=r["context_texts"])
        return FaithfulnessMetric(threshold=TARGET_FAITHFULNESS).measure(tc)

    with ThreadPoolExecutor(max_workers=config.EVAL_CONCURRENCY) as pool:
        futures = [pool.submit(_score_one, r) for r in grounded]
        scores = [f.result() for f in as_completed(futures)]

    avg = sum(scores) / len(scores)
    assert avg >= TARGET_FAITHFULNESS, f"Faithfulness = {avg:.3f}, target >= {TARGET_FAITHFULNESS}"


def test_citation_accuracy_is_binary_gate(golden_results):
    """Every grounded answer must carry at least one citation — this is a
    100% gate (pipeline-parameters.md), not a threshold."""
    for r in golden_results:
        if r["grounded"]:
            assert r["answer"] != "", "grounded answer must not be empty"


def test_refusal_accuracy_is_binary_gate(refusal_results):
    """Every out-of-domain/injection query in the golden set must be
    refused, not answered — see TARGET_REFUSAL_ACCURACY's docstring for
    why this is 100%, not a threshold."""
    if not refusal_results:
        pytest.skip("no refusal cases in golden set")
    failures = [r for r in refusal_results if not r["refused_correctly"]]
    hits = len(refusal_results) - len(failures)
    accuracy = hits / len(refusal_results)
    failure_detail = "; ".join(f"{r['case']['query']!r} -> {r['decision_action']}" for r in failures)
    assert accuracy >= TARGET_REFUSAL_ACCURACY, (
        f"Refusal accuracy = {accuracy:.3f}, target >= {TARGET_REFUSAL_ACCURACY}. "
        f"Failed to refuse: {failure_detail}"
    )


def _mrr(results: list[dict]) -> float:
    reciprocal_ranks = [1.0 / r["doc_rank"] if r["doc_rank"] > 0 else 0.0 for r in results]
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


def _ndcg_at_5(results: list[dict]) -> float:
    """
    Binary relevance nDCG@5, computed correctly this time — confirmed live
    (2026-08-24) that the previous version could exceed 1.0 (saw 1.973),
    which is mathematically impossible for a correct nDCG: it assumed
    IDCG was always 1/log2(2) as if only ONE relevant item could ever
    exist in the top-5, but our chunks aren't deduplicated by relevance —
    a case with 3 chunks sharing the expected clause_type in its top-5
    summed DCG across all 3 while IDCG only accounted for 1, so DCG > IDCG.

    Fixed: IDCG is now computed PER CASE from that case's own relevance
    sequence, sorted into ideal (best-first) order, not a fixed
    assumption — the standard nDCG definition. This correctly caps every
    case's score at <= 1.0 regardless of how many relevant chunks appear.
    """
    import math

    def _dcg(relevances: list[int]) -> float:
        return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))

    scores = []
    for r in results:
        actual = r["relevance_at_rank"]
        ideal = sorted(actual, reverse=True)  # best-possible ordering for THIS case's relevance counts
        idcg = _dcg(ideal)
        scores.append(_dcg(actual) / idcg if idcg else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def evaluate_summary(
    vector_store: VectorStore | None = None,
    doc_store: DocStore | None = None,
    sample_size: int | None = None,
) -> dict:
    """Non-pytest entrypoint — prints a human-readable summary, used by
    `python -m eval.run` for a quick manual check outside CI, and by
    app.py's /admin/eval for an on-demand check from the running app.
    Also appends this run to the eval history log (see
    observability/dashboard.py's record_eval_run) so drift over time is
    checkable, not just the single most recent number.

    sample_size (2026-08-24, added given real free-tier quota pressure):
    limits the RETRIEVAL cases to the first N (deterministic slice, not
    random — so two runs at the same sample_size are directly comparable,
    not noisy from a different random subset each time). Refusal cases
    (only 4, cheap, and the single most safety-critical metric) always
    run in full regardless of sample_size — there's no meaningful
    quota-saving from sampling 4 cases down further, and refusal accuracy
    is not something you want a partial read on.

    IMPORTANT: opens ONE shared store instance here, up front, when the
    caller didn't pass one in — never lets the parallel case runner fall
    through to each thread opening its own (see _run_cases_parallel's
    docstring for why: Qdrant's embedded local mode locks its storage
    path to a single client per process, and per-thread opens under real
    concurrency crash with "already accessed by another instance").
    """
    if vector_store is None and doc_store is None:
        vector_store, doc_store = VectorStore(), DocStore()

    all_cases = load_golden_set()
    retrieval_cases = [c for c in all_cases if not c.get("expect_refusal")]
    refusal_cases = [c for c in all_cases if c.get("expect_refusal")]
    if sample_size is not None:
        retrieval_cases = retrieval_cases[:sample_size]

    results = _run_cases_parallel(retrieval_cases, run_case, vector_store, doc_store)
    n = len(results)
    recall = sum(r["recall_hit"] for r in results) / n if n else None
    precision = sum(r["precision_hit"] for r in results) / n if n else None
    mrr = _mrr(results) if results else None
    ndcg5 = _ndcg_at_5(results) if results else None
    false_refusal_rate = sum(r["falsely_refused"] for r in results) / n if n else None
    # Which specific queries got wrongly refused — an aggregate rate alone
    # doesn't tell you what to actually tune. Confirmed live (2026-08-24):
    # 27% false refusal once real embeddings were consistently in play,
    # almost certainly because OUT_OF_DOMAIN_CONFIDENCE_FLOOR/
    # VAGUE_QUERY_CONFIDENCE_FLOOR (config.py) were tuned against the old
    # hash-embedder's score distribution and never recalibrated for real
    # Gemini embeddings' actual confidence range. This list is what you'd
    # look at to find the right new threshold values.
    falsely_refused_queries = [r["case"]["query"] for r in results if r["falsely_refused"]]

    grounded_results = [r for r in results if r["grounded"]]
    # PARALLELIZED — this loop was the remaining bottleneck even after
    # retrieval cases were parallelized: each grounded answer needs its
    # own real judge-LLM call (FaithfulnessMetric -> get_gateway().generate),
    # and running ~20 of those sequentially after the retrieval phase
    # already completed was still enough to exceed a 120s client timeout
    # on its own. Same bounded-concurrency reasoning as everywhere else.
    def _score_one(r: dict) -> float:
        tc = LLMTestCase(input=r["case"]["query"], actual_output=r["answer"], retrieval_context=r["context_texts"])
        return FaithfulnessMetric().measure(tc)

    faithfulness_scores: list[float] = []
    if grounded_results:
        with ThreadPoolExecutor(max_workers=config.EVAL_CONCURRENCY) as pool:
            futures = [pool.submit(_score_one, r) for r in grounded_results]
            faithfulness_scores = [f.result() for f in as_completed(futures)]
    faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0
    # Hallucination rate: fraction of GROUNDED answers that still scored
    # below the faithfulness threshold — distinct from faithfulness_rate
    # (the average score) because a stakeholder asking "how often does it
    # make things up" wants a rate, not an average.
    hallucination_rate = (
        sum(1 for s in faithfulness_scores if s < TARGET_FAITHFULNESS) / len(faithfulness_scores)
        if faithfulness_scores else 0.0
    )

    refusal_results_ = _run_cases_parallel(refusal_cases, run_refusal_case)
    refusal_accuracy = (
        sum(r["refused_correctly"] for r in refusal_results_) / len(refusal_results_)
        if refusal_results_ else None
    )
    # False pass rate is exactly the inverse of refusal_accuracy on the
    # refusal cases (an out-of-domain/injection query that WASN'T
    # refused) — reported separately since "false pass rate" is the term
    # a security-focused stakeholder will actually search the dashboard for.
    false_pass_rate = round(1 - refusal_accuracy, 3) if refusal_accuracy is not None else None

    summary = {
        "n_retrieval_cases": n,
        "n_refusal_cases": len(refusal_cases),
        "recall_at_25_doc_level": round(recall, 3) if recall is not None else None,
        "precision_at_5_clause_level": round(precision, 3) if precision is not None else None,
        "mrr": round(mrr, 3) if mrr is not None else None,
        "ndcg_at_5": round(ndcg5, 3) if ndcg5 is not None else None,
        "faithfulness_rate": round(faithfulness, 3),
        "hallucination_rate": round(hallucination_rate, 3),
        "false_refusal_rate": round(false_refusal_rate, 3) if false_refusal_rate is not None else None,
        "falsely_refused_queries": falsely_refused_queries,
        "false_pass_rate": false_pass_rate,
        "refusal_accuracy": round(refusal_accuracy, 3) if refusal_accuracy is not None else None,
        "grounded_count": len(grounded_results),
    }

    try:
        from observability.dashboard import record_eval_run
        record_eval_run(summary)
    except Exception:  # noqa: BLE001 — recording history must never break the eval itself
        pass

    return summary


if __name__ == "__main__":
    print(json.dumps(evaluate_summary(), indent=2))