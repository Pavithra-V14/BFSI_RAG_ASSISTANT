"""
Multi-turn conversation context — the standard RAG "query contextualization"
pattern: given the last few turns of a conversation plus a new (possibly
ambiguous) follow-up question, rewrite it into a standalone question that
retrieval can act on without needing the prior turns.

2026-08-24 — removed the keyword-based needs_contextualization() gate that
used to decide WHETHER to attempt a rewrite. That gate was a hardcoded
list of trigger words/phrases ("instead", "previous", "elaborate", ...) —
found and patched twice in one day after real phrasings slipped through
it uncaught (confirmed live: "give me more about the previous question"
silently skipped contextualization because "previous" wasn't on the list
yet). A keyword list is structurally incapable of covering every way a
person might phrase a follow-up; the next missed phrase was always just
a matter of time.

Production-standard fix, not another patch: this now ALWAYS attempts the
rewrite whenever prior turns exist, and lets the LLM itself judge whether
the query needs rewriting — the prompt already instructs it to return the
query completely unchanged if it's already standalone. This is the same
pattern LangChain's documented "condense question" chain uses. The only
gate left is structural (is there history at all, is a real provider
configured) — never a guess about the CONTENT of the query.

Why this is separate from guardrails/input_guardrail.py's existing
_rewrite_vague_query(): that function handles a DIFFERENT problem — a
first-turn query that's vague on its own (e.g. "waiting period", no
context to draw from). This module handles a follow-up that's only vague
BECAUSE it depends on prior turns (e.g. "what about for accidents
instead?" only makes sense after a preceding question about waiting
periods). Both can apply to the same query; contextualization runs first
in app.py's /query, so a follow-up gets "dereferenced" against history
before the vague-query rewrite (which has no history to draw from) would
otherwise have to guess.

This is SESSION memory (see ARCHITECTURE doc's memory-types discussion),
not Mem0. It only ever looks at the current session's own recent turns,
never another session's or another user's — same reasoning as everything
else in this codebase that's scoped by ownership.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 3  # last 3 user+assistant exchanges — bounds token cost,
                        # see the "context is a multiplier, not a constant"
                        # point from the architecture discussion


def _recent_turns(messages: list[dict], max_turns: int = MAX_HISTORY_TURNS) -> list[dict]:
    """Last N user/assistant exchanges, most recent last, oldest dropped —
    this IS the compression point: older history is truncated, not
    summarized, since session conversations here are short by nature
    (a compliance Q&A session, not a long-running chat)."""
    return messages[-(max_turns * 2):] if messages else []


def _format_history(turns: list[dict]) -> str:
    lines = []
    for m in turns:
        speaker = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {m['content']}")
    return "\n".join(lines)


def contextualize_query(query: str, session_messages: list[dict]) -> str:
    """
    Rewrites `query` into a standalone question using recent session
    history. Runs whenever there IS history — no content-based guessing
    about whether the query "looks like" it needs rewriting (see module
    docstring for why that approach was removed). The LLM itself decides,
    via the prompt below, whether the query needs to change at all.

    Falls back to returning the original query unchanged whenever:
      - there's no prior history (first turn in the session) — nothing to
        rewrite against, and no LLM call is made in this case, so this
        remains cheap for the common single-turn case
      - no real LLM provider is configured — LLMGateway's offline mock
        generator is built for Q&A sentence-extraction against retrieved
        chunks, NOT for rewriting a question; using it here would produce
        nonsense (e.g. returning a fragment of the conversation history
        verbatim instead of a rewritten question), so this is checked
        explicitly rather than relying on the mock's output happening to
        look reasonable
      - the LLM rewrite call fails for any reason (never block a query
        on this — worst case, retrieval runs on the un-rewritten query,
        same behavior as before this feature existed)
    """
    turns = _recent_turns(session_messages)
    if not turns:
        return query
    if not (os.environ.get("GROQ_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        logger.info("No LLM provider configured — skipping contextualization, using raw query")
        return query

    try:
        from gateway.llm_gateway import get_gateway
        gateway = get_gateway()
        history_text = _format_history(turns)
        system_prompt = (
            "Given the conversation history and a follow-up message, decide whether "
            "the follow-up needs the history to be understood. If it does, rewrite it "
            "as a standalone question that captures the same meaning without needing "
            "the history. If the follow-up is ALREADY a complete, standalone question "
            "on its own, return it completely UNCHANGED — do not paraphrase, do not "
            "\"improve\" it, only rewrite when the history is genuinely needed to "
            "understand what's being asked. Never answer the question, only rewrite "
            "or return it. Respond with ONLY the resulting question, nothing else."
        )
        result = gateway.generate(
            system_prompt=system_prompt,
            context_chunks=[f"Conversation history:\n{history_text}"],
            query=f"Follow-up message: {query}",
        )
        from observability.tracer import log_gateway_attempt
        log_gateway_attempt("context_rewrite_generation", result)
        rewritten = result["text"].strip().strip('"')
        # Sanity floor: a rewrite that's empty or wildly longer than
        # reasonable is more likely a malformed generation than a real
        # rewrite — fall back rather than risk feeding garbage to retrieval.
        if rewritten and len(rewritten) < 300:
            return rewritten
        logger.warning("Contextualization produced an unusable rewrite, using original query")
    except Exception as e:  # noqa: BLE001 — must never block a query on this
        logger.warning("Contextualization failed (%s), using original query", str(e)[:200])

    return query