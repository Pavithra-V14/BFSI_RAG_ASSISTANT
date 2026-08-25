"""
The harness — the actual product (per the architecture discussion). Every
route is thin; the decision logic lives in the modules it calls, testable
without hitting a real LLM.

Endpoints:
  POST /auth/register        create a user (self-serve role selection for
                              demo purposes — production: admin-provisioned)
  POST /auth/login            OAuth2 password flow -> JWT
  GET  /auth/me                current user's identity
  POST /sessions               create a chat session
  GET  /sessions                list the caller's own sessions
  GET  /sessions/{id}/messages  full transcript of one of the caller's sessions
  DELETE /sessions/{id}         delete one of the caller's sessions
  POST /query                   the RAG pipeline itself (auth required)
  POST /ingest                  document ingestion (compliance_officer/admin only)
  POST /feedback                thumbs up/down on an answer, for the eval loop
  GET  /admin/stats             role-gated operational snapshot (admin only)
  GET  /health                  liveness, no auth

Flow inside /query (three-layer guardrail stack from ADR 0005):
  input guardrail -> cache check -> retrieval -> rerank ->
  CONTEXT GUARDRAIL -> generation (via gateway) -> output guardrail ->
  cache write-back -> session transcript write -> trace flush
  (Mem0/long-term memory removed 2026-08-25 — confirmed it was fetched,
  counted, logged, and never actually used downstream; pure overhead with
  no functional effect on any response, and it also never survived a
  Render restart since it lived in a local-only embedded Qdrant store,
  unlike everything else in this pipeline which now persists to Postgres.)
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()  # MUST run before any module below reads os.environ at import
# time (config.py, gateway/llm_gateway.py's PROVIDER_CONFIG, etc. all
# resolve env vars when imported, not per-request) — without this call,
# editing .env has no effect at all; the app only sees whatever was
# already in the shell's environment when it was launched.

logger = logging.getLogger(__name__)

import jwt
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

import auth.store as store
import config
import sessions.store as session_store
from auth.deps import CurrentUser, get_current_user, require_role
from auth.security import create_access_token, hash_password, verify_password
from gateway.llm_gateway import get_gateway
from guardrails.context_guardrail import check_retrieved_context
from guardrails.input_guardrail import check_input
from guardrails.output_guardrail import apply_output_guardrail
from ingestion.run import deprecate_document, run_ingestion
from ingestion.store import DocStore, VectorStore
from observability import dashboard
from observability.tracer import read_recent_traces, start_trace
from retrieval.cache import SemanticCache
from retrieval.contextualize import contextualize_query
from retrieval.index import Candidate, hybrid_retrieve
from retrieval.llamaindex_retriever import BFSIHybridRetriever


def _rate_limit_key(request: Request) -> str:
    """
    Keys by the caller's bearer token when present (per-user limiting on
    an authenticated API — two different users must never share a quota),
    falling back to remote address for unauthenticated routes (login,
    register) where there's no token yet to key on. Deliberately does NOT
    decode/verify the JWT here — an opaque token string is enough to
    distinguish callers for rate-limiting purposes, and re-verifying it
    would duplicate auth.deps.get_current_user's job for no benefit.
    """
    auth_header = request.headers.get("authorization")
    if auth_header:
        return auth_header
    return get_remote_address(request)


limiter = Limiter(
    key_func=_rate_limit_key,
    storage_uri=config.rate_limit_storage_uri(),
    enabled=config.RATE_LIMIT_ENABLED,
)

app = FastAPI(title="BFSI RAG Assistant", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ALLOWED_ORIGINS,  # env: CORS_ALLOWED_ORIGINS, comma-separated;
    allow_methods=["*"],                          # defaults to just the local Streamlit dev origin
    allow_headers=["*"],                          # — never "*" outside local dev
    allow_credentials=True,
)


@app.on_event("startup")
def _start_rotation_scheduler() -> None:
    """
    Item 1.7 — real scheduling, not just the manual /admin/audit-log/rotate
    endpoint. Runs rotate_old_traces() automatically on an interval, as a
    background daemon thread. See config.AUDIT_LOG_ROTATION_CHECK_INTERVAL_
    SECONDS's docstring for the honest limitation: this only fires while
    the process is alive, which on Render's free tier isn't guaranteed
    (the instance sleeps after 15 min idle) — real production reliability
    needs an external scheduler, not just this. Still real, non-zero
    value for any deployment where the process stays up.
    """
    import threading

    def _rotation_loop():
        from observability.tracer import rotate_old_traces
        while True:
            time.sleep(config.AUDIT_LOG_ROTATION_CHECK_INTERVAL_SECONDS)
            try:
                result = rotate_old_traces()
                if result.get("rotated"):
                    logger.info("Scheduled rotation: %s", result)
            except Exception as e:  # noqa: BLE001 — a failed scheduled rotation
                logger.warning("Scheduled rotation failed (%s)", str(e)[:200])  # must never crash the app

    thread = threading.Thread(target=_rotation_loop, daemon=True, name="audit-log-rotation-scheduler")
    thread.start()

_vector_store, _doc_store = VectorStore(), DocStore()
_cache = SemanticCache()
_gateway = get_gateway()  # shared singleton — see gateway/llm_gateway.py

INGEST_ROLES = ("compliance_officer", "admin")
ADMIN_ROLES = ("admin",)


# ─────────────────────────── schemas ───────────────────────────

class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    email: str
    password: str = Field(min_length=8)
    # No `role` field — self-serve registration always creates the
    # lowest-privilege role (auth_store.SELF_SERVE_ROLES). Anything higher
    # requires an existing admin to promote via PATCH /admin/users/{id}/role.
    # This closes the previously-open gap where anyone could self-register
    # as admin by just typing "role": "admin" in the request body.


class ForgotPasswordRequest(BaseModel):
    email: str


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str = Field(min_length=8)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    username: str


class MeResponse(BaseModel):
    username: str
    role: str
    user_id: int


class SessionResponse(BaseModel):
    session_id: str
    title: str
    created_at: str


class QueryRequest(BaseModel):
    query: str
    session_id: str | None = None
    product_line: str | None = None


class QueryResponse(BaseModel):
    answer: str
    grounded: bool
    citations: list[str]
    trace_id: str
    decision: str
    session_id: str
    degraded: bool = False       # 2026-08-24 — item 5: true when the answer came
    generation_provider: str | None = None  # from the offline mock, not a real LLM.
    # Before this, a mock-generated answer and a real one were structurally
    # identical in the response — a compliance officer had no way to know
    # whether a real model reasoned through the answer or a sentence-
    # extraction fallback stitched it together. Confirmed as a real gap
    # during today's provider-outage testing.


class IngestRequest(BaseModel):
    doc_path: str
    source: str
    product_line: str = "health"
    access_role: list[str] | None = None
    dry_run: bool = False


class FeedbackRequest(BaseModel):
    trace_id: str
    session_id: str
    rating: str  # "up" | "down"
    comment: str | None = None


class RoleUpdateRequest(BaseModel):
    role: str
    confirm_password: str  # 2026-08-24 — elevated auth: promoting someone
    # to admin/compliance_officer must re-confirm the ACTING admin's own
    # password, not just rely on the JWT already in hand. A leaked admin
    # token alone should not be enough to mint more admins — this closes
    # that gap by requiring a second factor (something the attacker would
    # also need to know, not just possess).


class UserSummary(BaseModel):
    id: int
    username: str
    email: str
    role: str
    created_at: str


# ─────────────────────────── auth ───────────────────────────

@app.post("/auth/register", response_model=TokenResponse, status_code=201)
def register(req: RegisterRequest) -> TokenResponse:
    if store.username_or_email_exists(req.username, req.email):
        raise HTTPException(409, "username or email already registered")
    # Self-serve always gets the lowest-privilege role — see RegisterRequest's
    # docstring. EXCEPTION: on a genuinely fresh deployment (zero admins
    # exist yet), the first successful registration is auto-promoted to
    # admin — see auth_store.promote_if_first_user()'s docstring for the
    # race-safety guarantee. This is the alternative bootstrap path to
    # scripts/create_admin.py's shell-access script; set
    # AUTO_PROMOTE_FIRST_USER_TO_ADMIN=false to require that script
    # instead, if you'd rather admin creation stay gated behind actual
    # server access every time.
    role = next(iter(store.SELF_SERVE_ROLES))
    user = store.create_user(req.username, req.email, hash_password(req.password), role)

    if store.promote_if_first_user(user.id):
        logger.warning(
            "AUTO-PROMOTED to admin: username=%s (first user on this deployment — "
            "no prior admin existed). Set AUTO_PROMOTE_FIRST_USER_TO_ADMIN=false to disable this.",
            user.username,
        )
        user = store.User(id=user.id, username=user.username, email=user.email, role="admin")

    token = create_access_token(subject=user.username, role=user.role)
    return TokenResponse(access_token=token, role=user.role, username=user.username)


@app.post("/auth/login", response_model=TokenResponse)
def login(form: OAuth2PasswordRequestForm = Depends()) -> TokenResponse:
    if store.is_locked_out(form.username):
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"account temporarily locked after {store.MAX_FAILED_LOGIN_ATTEMPTS} failed attempts — "
            f"try again in up to {store.LOCKOUT_DURATION_SECONDS // 60} minutes",
        )
    row = store.get_user_by_username(form.username)
    if row is None or not verify_password(form.password, row["password_hash"]):
        store.record_failed_login(form.username)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "incorrect username or password")
    store.record_successful_login(row["id"])
    token = create_access_token(subject=row["username"], role=row["role"])
    return TokenResponse(access_token=token, role=row["role"], username=row["username"])


@app.post("/auth/logout")
def logout(request: Request, user: CurrentUser = Depends(get_current_user)) -> dict:
    """
    Token revocation (item 1) — without this, a JWT is valid until it
    naturally expires (24h default) no matter what; a leaked or
    no-longer-wanted token had no way to be killed early. Revokes ONLY
    the current token (by its jti), not every token this user holds —
    logging out on one device shouldn't force-logout every other session.
    """
    if user.jti:
        payload = decode_access_token_from_request(request)
        exp = payload.get("exp", time.time() + 86400) if payload else time.time() + 86400
        store.revoke_token(user.jti, exp)
    return {"logged_out": True}


def decode_access_token_from_request(request: Request) -> dict | None:
    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        return None
    try:
        return decode_access_token(auth_header[7:])
    except Exception:  # noqa: BLE001
        return None


def _send_password_reset_email(to_email: str, username: str, token: str) -> bool:
    """
    Real email delivery via Resend (2026-08-25) — replaces the earlier
    dev-mode stopgap. Returns True if actually sent, False if it fell
    back to console logging (no RESEND_API_KEY configured, or the API
    call itself failed) — the caller doesn't change behavior either way
    (same generic response to the user regardless), this return value is
    only used for the log message's wording.
    """
    reset_link = f"{config.FRONTEND_BASE_URL}/?reset_token={token}"
    if not config.RESEND_API_KEY:
        logger.warning(
            "PASSWORD RESET (RESEND_API_KEY not configured — logging instead of emailing): "
            "user=%s reset_link=%s (valid %d minutes). Set RESEND_API_KEY to send real emails — "
            "see config.py's Email delivery section.",
            username, reset_link, store.PASSWORD_RESET_TOKEN_TTL_SECONDS // 60,
        )
        return False

    try:
        import resend
        resend.api_key = config.RESEND_API_KEY
        # 2026-08-25 — removed a debug print() that was here logging the
        # REAL Resend API key to console/logs on every send attempt
        # (`print(f"Sending password reset email to {to_email}", resend.api_key)`).
        # A real secret landing in plaintext logs is a genuine security
        # issue — Render's log viewer, or anyone with log access, would
        # have been able to read your live API key directly. Never log
        # API keys/secrets, even for debugging; log that a send was
        # attempted, not the credential used to do it.
        resend.Emails.send({
            "from": config.RESEND_FROM_ADDRESS,
            "to": to_email,
            "subject": "Reset your BFSI RAG Assistant password",
            "html": (
                f"<p>Hi {username},</p>"
                f"<p>Click below to reset your password. This link expires in "
                f"{store.PASSWORD_RESET_TOKEN_TTL_SECONDS // 60} minutes.</p>"
                f'<p><a href="{reset_link}">{reset_link}</a></p>'
                f"<p>If you didn't request this, you can safely ignore this email.</p>"
            ),
        })
        return True
    except Exception as e:  # noqa: BLE001 — a failed send must not crash the request or
        logger.warning(  # leak whether the email exists via a differing error response
            "Password reset email failed to send (%s) — falling back to console log. "
            "user=%s reset_link=%s", str(e)[:200], username, reset_link,
        )
        return False


@app.post("/auth/forgot-password")
def forgot_password(req: ForgotPasswordRequest) -> dict:
    """
    Sends a real password-reset email via Resend when RESEND_API_KEY is
    configured (config.py) — falls back to logging the reset link
    server-side otherwise, same graceful-degradation pattern as every
    other optional integration in this codebase. Always returns the same
    generic response regardless of whether the email exists or whether
    sending succeeded, to avoid leaking which emails are registered or
    whether email delivery is even configured.
    """
    row = _get_user_by_email(req.email)
    if row is not None:
        token = store.create_password_reset_token(row["id"])
        _send_password_reset_email(row["email"], row["username"], token)
    return {"message": "If that email is registered, a reset link has been sent."}


def _get_user_by_email(email: str):
    for row in store.list_users():
        if row["email"] == email:
            return store.get_user_by_username(row["username"])
    return None


@app.post("/auth/reset-password")
def reset_password(req: ResetPasswordRequest) -> dict:
    user_id = store.consume_password_reset_token(req.token)
    if user_id is None:
        raise HTTPException(400, "invalid, expired, or already-used reset token")
    store.update_password(user_id, hash_password(req.new_password))
    return {"password_reset": True}


@app.get("/auth/me", response_model=MeResponse)
def me(user: CurrentUser = Depends(get_current_user)) -> MeResponse:
    return MeResponse(username=user.username, role=user.role, user_id=user.user_id)


# ─────────────────────────── sessions ───────────────────────────

@app.post("/sessions", response_model=SessionResponse, status_code=201)
def create_session(user: CurrentUser = Depends(get_current_user)) -> SessionResponse:
    s = session_store.create_session(user.user_id)
    return SessionResponse(session_id=s.session_id, title=s.title, created_at=s.created_at)


@app.get("/sessions", response_model=list[SessionResponse])
def list_sessions(user: CurrentUser = Depends(get_current_user)) -> list[SessionResponse]:
    return [SessionResponse(**s) for s in session_store.list_sessions(user.user_id)]


@app.get("/sessions/{session_id}/messages")
def get_messages(session_id: str, user: CurrentUser = Depends(get_current_user)) -> list[dict]:
    return session_store.get_messages(session_id, user.user_id)


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str, user: CurrentUser = Depends(get_current_user)) -> dict:
    ok = session_store.delete_session(session_id, user.user_id)
    if not ok:
        raise HTTPException(404, "session not found or not owned by caller")
    return {"deleted": True}


# ─────────────────────────── ingestion (authorized roles only) ───────────────────────────

@app.post("/ingest")
@limiter.limit(f"{config.RATE_LIMIT_INGEST_PER_MINUTE}/minute")
def ingest(request: Request, req: IngestRequest, user: CurrentUser = Depends(require_role(*INGEST_ROLES))) -> dict:
    result = run_ingestion(
        req.doc_path, req.source, req.product_line, req.dry_run, req.access_role,
        vector_store=_vector_store, doc_store=_doc_store,
    )
    for meta in result.get("sample_metadata", []):
        if meta.get("version", 1) > 1:
            _cache.invalidate_for_chunk(meta["chunk_id"], meta["version"])
    return result


@app.delete("/documents/{doc_id}")
def retract_document(doc_id: str, user: CurrentUser = Depends(require_role(*INGEST_ROLES))) -> dict:
    """
    Withdraws a document entirely — for a regulation that's fully
    superseded with no replacement text, not for an amendment (which is
    just re-running /ingest on the updated file). Soft-closes every chunk
    via effective_to, same mechanism M2 already uses; chunks stay
    queryable with include_superseded=True for audit replay, they just
    stop being retrieved for normal queries.
    """
    result = deprecate_document(doc_id, vector_store=_vector_store, doc_store=_doc_store)
    if result["status"] == "not_found":
        raise HTTPException(404, f"no chunks found for doc_id '{doc_id}'")
    return result


# ─────────────────────────── the RAG pipeline ───────────────────────────

@app.post("/query", response_model=QueryResponse)
@limiter.limit(f"{config.RATE_LIMIT_QUERY_PER_MINUTE}/minute")
def query(request: Request, req: QueryRequest, user: CurrentUser = Depends(get_current_user)) -> QueryResponse:
    is_continuing_session = req.session_id is not None
    session_id = req.session_id or session_store.create_session(user.user_id).session_id
    if not session_store.get_session(session_id, user.user_id):
        raise HTTPException(404, "session not found or not owned by caller")

    # 2026-08-24 fix: PII masking previously only ever scanned RETRIEVED
    # document chunks and the GENERATED answer — never the user's own raw
    # query. Confirmed live: a query containing a real name and email
    # ("My name is Pavithra V & my email id is pavithra@gmail.com...")
    # got stored completely unmasked in the session transcript and the
    # session TITLE (visible in the Streamlit sidebar) — with the output
    # guardrail correctly but misleadingly reporting "no PII found" (true
    # for the context/answer it checks, silent about the query itself).
    # masked_query_for_storage is used at every point the query gets
    # PERSISTED below; the REAL req.query is still used for
    # contextualization/retrieval/generation unmasked — masking wouldn't
    # help the model answer and could garble a legitimate question that
    # happens to overlap PII-shaped text.
    from guardrails.pii import mask_pii as _mask_pii_for_storage
    masked_query_for_storage, query_pii_found, query_pii_types = _mask_pii_for_storage(req.query)

    with start_trace("query", user_id=str(user.user_id)) as trace:

        # 0. multi-turn contextualization — rewrite a bare follow-up
        # ("what about for accidents instead?") into a standalone question
        # using this session's own recent history, BEFORE it reaches the
        # guardrail probe or retrieval. Only fetches history for an
        # EXISTING session — a brand-new session has no prior turns, so
        # skip the store call entirely rather than fetching an empty list.
        prior_messages = (
            session_store.get_messages(session_id, user.user_id) if is_continuing_session else []
        )
        contextualized_query = contextualize_query(req.query, prior_messages)
        trace.log_stage(
            "context_rewrite",
            rewritten=contextualized_query != req.query,
            history_turns=len(prior_messages),
        )

        # 1. probe + input guardrail (LLM Guard, with regex fallback) —
        # runs on the CONTEXTUALIZED query, so a short in-domain follow-up
        # doesn't get misread as vague/out-of-domain the way the bare
        # follow-up alone would.
        probe = hybrid_retrieve(
            contextualized_query, _vector_store, _doc_store,
            filters={"access_role": [user.role, "*"]}, top_k=5,
        )
        probe_confidence = probe[0].score if probe else 0.0
        decision = check_input(contextualized_query, retrieval_confidence=probe_confidence)
        trace.log_stage(
            "input_guardrail", action=decision.action, reason=decision.reason,
            query_pii_found=query_pii_found, query_pii_detected_types=query_pii_types,
        )

        if decision.action in ("refuse", "block"):
            refusal = (
                "I can't help with that."
                if decision.action == "block"
                else "I'm scoped to BFSI compliance and claims questions — that looks outside my scope."
            )
            trace.log_stage("output_guardrail", grounded=False, short_circuit=True)
            session_store.add_message(session_id, "user", masked_query_for_storage)
            session_store.add_message(session_id, "assistant", refusal, grounded=False)
            return QueryResponse(answer=refusal, grounded=False, citations=[], trace_id=trace.trace_id, decision=decision.action, session_id=session_id)

        effective_query = decision.query
        # Log the RAW message the user actually typed, not the rewritten
        # standalone version — the transcript should read naturally.
        # PII-masked before storage (see masked_query_for_storage above).
        session_store.add_message(session_id, "user", masked_query_for_storage)
        session_store.rename_session_if_first_message(session_id, user.user_id, masked_query_for_storage)

        # 2. cache check (Valkey, scoped by ROLE — see retrieval/cache.py's
        # 2026-08-23 fix docstring for why role, not user_id or a content
        # guess, is the correct scoping dimension: role is exactly what
        # determines document access, so same-role callers safely share a
        # cache entry and cross-role callers never do.)
        cache_hit = _cache.lookup(effective_query, role=user.role)
        trace.log_stage("cache_check", hit=cache_hit is not None, role=user.role)
        if cache_hit:
            cached_provider = cache_hit.get("generation_provider")
            cached_degraded = cached_provider in ("offline_mock", "offline_mock_fallback", None)
            trace.log_stage("generation", cached=True, provider=cached_provider)
            trace.log_stage("output_guardrail", grounded=True, cached=True)
            session_store.add_message(session_id, "assistant", cache_hit["answer"], citations=cache_hit["citations"], grounded=True)
            return QueryResponse(
                answer=cache_hit["answer"], grounded=True, citations=cache_hit["citations"],
                trace_id=trace.trace_id, decision="cache_hit", session_id=session_id,
                degraded=cached_degraded, generation_provider=cached_provider,
            )

        # 3. retrieval + rerank — via the LlamaIndex-native retriever
        # wrapper (see ADR 0001 / retrieval/llamaindex_retriever.py), not
        # a direct hybrid_retrieve()+rerank() call. Same underlying Qdrant
        # hybrid search and reranker either way — this just composes them
        # through LlamaIndex's BaseRetriever contract so the harness is
        # ready to plug in other LlamaIndex-native components (query
        # engines, response synthesizers) without another rewrite later.
        filters = {"access_role": [user.role, "*"]}
        if req.product_line:
            filters["product_line"] = req.product_line
        retriever = BFSIHybridRetriever(_vector_store, _doc_store, filters=filters)
        nodes = retriever.retrieve(effective_query)
        trace.log_stage("retrieval", candidate_count=retriever.last_candidate_count)

        if not nodes:
            answer = "I couldn't find this in the available documents. Please check with a compliance officer."
            trace.log_stage("output_guardrail", grounded=False, reason="empty_retrieval")
            session_store.add_message(session_id, "assistant", answer, grounded=False)
            return QueryResponse(answer=answer, grounded=False, citations=[], trace_id=trace.trace_id, decision="fail_closed", session_id=session_id)

        # Convert LlamaIndex's NodeWithScore back to our own Candidate type
        # — everything downstream (context_guardrail, citation formatting)
        # is written against that contract; converting here keeps this the
        # only place that needs to know a LlamaIndex retriever was used.
        reranked = [
            Candidate(chunk_id=n.node.id_, score=n.score, metadata=n.node.metadata, text=n.node.text)
            for n in nodes
        ]
        trace.log_stage("rerank", final_chunk_count=len(reranked))

        # 4. CONTEXT GUARDRAIL — between rerank and generation (ADR 0005)
        ctx_result = check_retrieved_context(reranked, user_role=user.role)
        trace.log_stage(
            "context_guardrail", passed=ctx_result.passed,
            dropped_count=len(ctx_result.dropped), pii_masked_count=len(ctx_result.pii_masked_chunk_ids),
            pii_detected_types=ctx_result.pii_detected_types,  # item 7 audit trail — categories only, never values
        )
        if not ctx_result.passed:
            answer = "I couldn't find sufficiently reliable information to answer this. Please check with a compliance officer."
            trace.log_stage("output_guardrail", grounded=False, reason=ctx_result.reason)
            session_store.add_message(session_id, "assistant", answer, grounded=False)
            return QueryResponse(answer=answer, grounded=False, citations=[], trace_id=trace.trace_id, decision="fail_closed", session_id=session_id)
        surviving = ctx_result.surviving_chunks

        # 5. generation via gateway (Groq -> Gemini fallback, or offline mock)
        context_texts = [c.text for c in surviving]
        gen_result = _gateway.generate(
            system_prompt="Answer only from the provided BFSI policy/regulatory context. Cite clauses. Be concise.",
            context_chunks=context_texts, query=effective_query,
        )
        trace.log_stage(
            "generation", provider=gen_result["provider"], error=gen_result["error"],
            estimated_cost=gen_result.get("estimated_cost", 0.0),
            provider_attempts=gen_result.get("provider_attempts", []),
        )

        # 6. output guardrail (Guardrails AI schema + groundedness score + PII mask)
        chunk_dicts = [{"chunk_id": c.chunk_id, "source": c.metadata.get("source"), "text": c.text} for c in surviving]
        out = apply_output_guardrail(gen_result["text"], chunk_dicts)

        # Belt-and-suspenders staleness check: context_guardrail already
        # drops any chunk with effective_to set (see guardrails/
        # context_guardrail.py), so this should structurally never fire —
        # logging it anyway turns "we assume this invariant holds" into
        # "we can prove it holds," which is the actual point of tracking
        # it on the dashboard rather than just trusting the code comment.
        stale_citation_count = sum(
            1 for c in surviving if c.metadata.get("effective_to")
        )
        trace.log_stage(
            "output_guardrail", grounded=out.grounded, pii_masked=out.pii_masked,
            schema_valid=out.schema_valid, stale_citation_count=stale_citation_count,
            pii_detected_types=out.pii_detected_types,
        )

        # 7. cache + session write-back
        if out.grounded:
            chunk_versions = {c.chunk_id: c.metadata.get("version", 1) for c in surviving}
            _cache.store(
                effective_query, user.role, out.text, out.citations, chunk_versions,
                generation_provider=gen_result["provider"],
            )
        session_store.add_message(session_id, "assistant", out.text, citations=out.citations, grounded=out.grounded)

        is_degraded = gen_result["provider"] in ("offline_mock", "offline_mock_fallback", None)
        from observability import alerting
        alerting.record_generation_outcome(degraded=is_degraded, provider=gen_result["provider"])
        return QueryResponse(
            answer=out.text, grounded=out.grounded, citations=out.citations,
            trace_id=trace.trace_id, decision=decision.action, session_id=session_id,
            degraded=is_degraded, generation_provider=gen_result["provider"],
        )


# ─────────────────────────── feedback (feeds the eval loop) ───────────────────────────

@app.post("/feedback")
def feedback(req: FeedbackRequest, user: CurrentUser = Depends(get_current_user)) -> dict:
    if req.rating not in ("up", "down"):
        raise HTTPException(400, "rating must be 'up' or 'down'")
    dashboard.record_feedback(req.trace_id, req.session_id, user.user_id, req.rating, req.comment)
    return {"recorded": True}


# ─────────────────────────── admin ───────────────────────────

@app.get("/admin/users", response_model=list[UserSummary])
def list_users(user: CurrentUser = Depends(require_role(*ADMIN_ROLES))) -> list[UserSummary]:
    return [UserSummary(**dict(row)) for row in store.list_users()]


@app.patch("/admin/users/{user_id}/role")
def update_user_role(
    user_id: int, req: RoleUpdateRequest, user: CurrentUser = Depends(require_role(*ADMIN_ROLES))
) -> dict:
    """
    The only way to grant compliance_officer or admin — closes the gap
    where /auth/register used to let anyone self-select any role.

    Elevated auth (item 4): requires the ACTING admin's own current
    password, re-verified here, not just the JWT already attached to the
    request. A stolen/leaked admin token alone is no longer sufficient to
    mint more admins — the attacker would also need the password.
    """
    acting_admin = store.get_user_by_username(user.username)
    if acting_admin is None or not verify_password(req.confirm_password, acting_admin["password_hash"]):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "password confirmation failed")

    if req.role not in store.VALID_ROLES:
        raise HTTPException(400, f"role must be one of {sorted(store.VALID_ROLES)}")
    target = store.get_user_by_id(user_id)
    if target is None:
        raise HTTPException(404, "user not found")
    store.update_role(user_id, req.role)
    return {"user_id": user_id, "username": target["username"], "new_role": req.role}


@app.get("/admin/stats")
def admin_stats(
    n: int = 200,
    request_type: str | None = None,
    user: CurrentUser = Depends(require_role(*ADMIN_ROLES)),
) -> dict:
    """
    The real dashboard — CPSO, latency percentiles, cache hit rate,
    guardrail trigger rates, cost — computed from the trace log, not the
    thin trace-completeness-only version this used to return. See
    observability/dashboard.py for how each number is derived.

    request_type: None = live + eval blended (original behavior).
    "query" = live traffic only. "eval" = evaluation runs only (item 9 —
    eval traces existed in the log all along, but were never separated
    from live traffic in the aggregate numbers until now).
    """
    traces = read_recent_traces(n * 3 if request_type else n)
    if request_type:
        traces = {
            tid: stages for tid, stages in traces.items()
            if stages and stages[0].get("request_type", "").startswith(request_type)
        }
    trace_n = len(traces)
    complete = sum(1 for stages in traces.values() if any(s["stage"] == "_trace_end" for s in stages))

    metrics = dashboard.as_dict(n, request_type_filter=request_type)
    metrics["trace_completeness_rate"] = round(complete / trace_n, 3) if trace_n else None
    metrics["request_type_filter"] = request_type
    from observability import alerting
    alerting.alert_on_dashboard_thresholds(metrics)
    return metrics


@app.post("/admin/eval")
def admin_eval(
    sample_size: int | None = None, user: CurrentUser = Depends(require_role(*ADMIN_ROLES))
) -> dict:
    """
    Runs the golden-set eval suite live against the current pipeline —
    real Qdrant retrieval, real reranker/generation if configured (costs
    real API calls if keys are set; uses the offline fallback path
    otherwise). This is the on-demand version of `pytest
    eval/test_golden_set.py` for when you want a fresh number from the
    running app rather than a separate terminal command. Every run is
    also appended to the eval history log (see /admin/eval-history).

    sample_size: run only the first N retrieval cases instead of all 22 —
    added given real free-tier quota pressure (Cohere's 10 calls/minute,
    Gemini's 20 requests/day) confirmed live to get exhausted by a single
    full run. All 4 refusal cases always run regardless — cheap, and the
    single most safety-critical metric, not something to sample down.

    2026-08-24 — sample_size=0 or negative used to silently do something
    confusing rather than error: confirmed live, sample_size=-1 ran 21 of
    22 cases (Python's list[:-1] semantics), not the "run 1 case" a
    reasonable person would expect from a negative number, and
    sample_size=0 silently ran zero retrieval cases with no explanation.
    Both now reject with a clear 400 instead.
    """
    if sample_size is not None and sample_size < 1:
        raise HTTPException(400, "sample_size must be a positive integer (or omitted to run the full set)")
    from eval.test_golden_set import evaluate_summary
    return evaluate_summary(vector_store=_vector_store, doc_store=_doc_store, sample_size=sample_size)


@app.get("/admin/eval-history")
def admin_eval_history(last_n: int = 20, user: CurrentUser = Depends(require_role(*ADMIN_ROLES))) -> list[dict]:
    """Drift check — is retrieval/faithfulness quality trending down over
    successive eval runs, not just what the single most recent run says."""
    return dashboard.read_eval_history(last_n)


@app.get("/admin/audit-log/verify")
def admin_verify_audit_log(user: CurrentUser = Depends(require_role(*ADMIN_ROLES))) -> dict:
    """Item 8 — tamper detection. Recomputes the hash chain over the
    active trace log and reports whether it's intact, and exactly where
    it breaks if not."""
    from observability.tracer import verify_chain
    return verify_chain()


@app.post("/admin/audit-log/rotate")
def admin_rotate_audit_log(
    retention_days: int | None = None, user: CurrentUser = Depends(require_role(*ADMIN_ROLES))
) -> dict:
    """Item 8 — retention. Moves entries older than retention_days
    (default: config.AUDIT_LOG_RETENTION_DAYS) into a dated archive file —
    never deletes them. Safe to call repeatedly / on a schedule."""
    from observability.tracer import rotate_old_traces
    return rotate_old_traces(retention_days)


@app.post("/admin/observability/reset")
def admin_reset_observability(
    confirm: bool = False, user: CurrentUser = Depends(require_role(*ADMIN_ROLES))
) -> dict:
    """
    Clears the dashboard's accumulated data (traces, eval history,
    feedback) for a genuinely fresh start — e.g. after a full day of
    heavy testing, when the numbers no longer reflect current reality.

    Deliberately NOT the same as manually deleting the files by hand:
    that would corrupt the trace log's hash chain (item 8) — the tracer
    keeps an in-memory cache of "the last hash" that a raw file delete
    doesn't know to reset, so the very next write would silently produce
    an invalid chain. This endpoint resets both the files AND the
    relevant in-memory state (tracer's chain-hash cache, alerting's
    consecutive-failure counter) consistently.

    Requires confirm=true — this is destructive and irreversible for the
    active log (archived files under data/traces_archive/ are untouched,
    since those represent already-properly-retained history, not
    something a routine "start fresh" action should erase).
    """
    if not confirm:
        raise HTTPException(
            400,
            "This clears all dashboard trace/eval/feedback data. Call again with "
            "?confirm=true to proceed. Archived files (data/traces_archive/) are NOT affected.",
        )

    from observability.tracer import reset_trace_log
    from observability import alerting

    trace_result = reset_trace_log()
    feedback_result = dashboard.reset_feedback_and_eval_history()
    alerting.reset_state_for_testing()

    return {"reset": True, "traces": trace_result, "cleared": feedback_result["cleared"]}


@app.get("/health")
def health() -> dict:
    """
    Actually checks dependency connectivity, unlike the placeholder this
    replaces (which returned {"status": "ok"} unconditionally — a broken
    Qdrant or database connection would only surface as a mysterious 500
    on the next real request, not here, which defeats the point of a
    liveness/readiness check).

    Returns 200 with per-dependency status either way — a degraded
    dependency (e.g. Valkey down, cache falls back to no-op) is reported
    as "degraded", not conflated with a hard failure, since several
    dependencies in this system have documented graceful fallbacks and
    "unreachable" doesn't always mean "the app can't serve requests."
    """
    checks: dict[str, str] = {}

    try:
        _vector_store.client.get_collections()
        checks["qdrant"] = "ok"
    except Exception as e:  # noqa: BLE001
        checks["qdrant"] = f"error: {str(e)[:100]}"

    try:
        if _cache._redis is not None:
            _cache._redis.ping()
            checks["valkey"] = "ok"
        else:
            checks["valkey"] = "degraded: no-op fallback active (see retrieval/cache.py)"
    except Exception as e:  # noqa: BLE001
        checks["valkey"] = f"error: {str(e)[:100]}"

    try:
        store.list_users()
        checks["database"] = "ok"
    except Exception as e:  # noqa: BLE001
        checks["database"] = f"error: {str(e)[:100]}"

    hard_failures = [k for k, v in checks.items() if v.startswith("error")]
    status_label = "unhealthy" if hard_failures else ("degraded" if any(v.startswith("degraded") for v in checks.values()) else "ok")
    return {"status": status_label, "checks": checks}