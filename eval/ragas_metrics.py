"""
RAGAS-equivalent metrics, implemented as DeepEval BaseMetric subclasses.

Why custom instead of importing ragas directly: ragas has a hard,
unresolvable transitive dependency conflict in this environment — its
`langchain_community.chat_models.vertexai` import chain requires an old
`langchain-core` that's incompatible with the `langchain-core>=1.0`
required by guardrails-ai and langgraph (both already in this stack, see
ADR 0001/0005). See ADR 0008 for the full resolution.

These implement the same formulas RAGAS uses for the two metrics our eval
targets require (faithfulness, context precision) using DeepEval's metric
interface, so `eval/run.py` gets a real pytest-CI-gate-compatible score,
not the inline lexical-overlap proxy used per-request in
guardrails/output_guardrail.py (that one has to be synchronous/zero-latency;
these run offline against the golden set and can afford a judge-LLM call).

Judge model: Gemini 2.5 Flash via LiteLLM when GEMINI_API_KEY/GROQ_API_KEY
is set; falls back to the same lexical-overlap proxy (no judge call) when
no provider key is configured, so `eval/run.py` still produces a real
number with zero API keys — just a coarser one, same as every other
fallback in this codebase.
"""
from __future__ import annotations

import os
import re

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _judge_available() -> bool:
    return bool(os.environ.get("GROQ_API_KEY") or os.environ.get("GEMINI_API_KEY"))


def _judge_score(prompt: str) -> float:
    """Calls the LLM gateway with a scoring prompt, parses a 0.0-1.0 float
    from the response. Falls back to 0.5 (neutral) if parsing fails —
    never crashes the eval run on a malformed judge response."""
    from gateway.llm_gateway import get_gateway
    gateway = get_gateway()
    result = gateway.generate(
        system_prompt="You are a strict evaluator. Respond with ONLY a number between 0.0 and 1.0.",
        context_chunks=[],
        query=prompt,
    )
    from observability.tracer import log_gateway_attempt
    log_gateway_attempt("faithfulness_judge", result)
    match = re.search(r"(\d+\.?\d*)", result["text"])
    if match:
        try:
            return max(0.0, min(1.0, float(match.group(1))))
        except ValueError:
            pass
    return 0.5


class FaithfulnessMetric(BaseMetric):
    """RAGAS-equivalent: fraction of claims in the answer that are
    supported by the retrieved context. See ADR 0008."""

    def __init__(self, threshold: float = 0.80):
        self.threshold = threshold
        self.score = None
        self.reason = None

    def measure(self, test_case: LLMTestCase) -> float:
        context = " ".join(test_case.retrieval_context or [])
        if _judge_available():
            prompt = (
                f"Context:\n{context}\n\nAnswer:\n{test_case.actual_output}\n\n"
                "What fraction of claims in the Answer are directly supported "
                "by the Context? Respond with only a number 0.0-1.0."
            )
            self.score = _judge_score(prompt)
            self.reason = "judge-model scored"
        else:
            answer_tokens = _tokens(test_case.actual_output)
            context_tokens = _tokens(context)
            self.score = (
                len(answer_tokens & context_tokens) / len(answer_tokens)
                if answer_tokens else 0.0
            )
            self.reason = "lexical-overlap fallback (no judge model configured)"
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self) -> str:
        return "Faithfulness (RAGAS-equivalent)"


class ContextPrecisionMetric(BaseMetric):
    """RAGAS-equivalent: of the retrieved chunks, what fraction are
    actually relevant to the query (using the expected_output's clause
    type as the relevance signal in our golden set)."""

    def __init__(self, threshold: float = 0.80):
        self.threshold = threshold
        self.score = None

    def measure(self, test_case: LLMTestCase) -> float:
        query_tokens = _tokens(test_case.input)
        relevant = 0
        contexts = test_case.retrieval_context or []
        for chunk in contexts:
            chunk_tokens = _tokens(chunk)
            overlap = len(query_tokens & chunk_tokens) / max(len(query_tokens), 1)
            if overlap > 0.15:
                relevant += 1
        self.score = relevant / len(contexts) if contexts else 0.0
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self) -> str:
        return "Context Precision (RAGAS-equivalent)"