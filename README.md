# BFSI RAG Assistant — full enterprise stack

FastAPI backend + Streamlit frontend + JWT auth/RBAC, on top of the full RAG
architecture from this project: **LlamaIndex** · **Qdrant** · **Cohere
Rerank 3** · **LiteLLM (Groq/Gemini)** · **LLM Guard + Guardrails AI**
(three-layer guardrails) · **Mem0** · **Valkey** · **Langfuse** ·
**DeepEval**.

Every framework is a real, installed dependency. Where a component needs an
API key or network access this sandbox doesn't have, it falls back
automatically to tested logic and logs a clear warning — see §6.

---

## 1. Install

```bash
cd bfsi-rag-assistant
pip install -r requirements.txt
cp .env.example .env
```

Requires Python 3.10+ and a running Valkey (or Redis-protocol-compatible)
server. Two ways to get one:

**Option A — local, self-hosted (default, what the demo used):**
```bash
apt-get install valkey-server valkey-tools
valkey-server --daemonize yes
```

**Option B — Aiven for Valkey (managed, useful if local install is blocked
— e.g. WSL systemd issues):**
1. Sign up at [aiven.io](https://aiven.io) (free tier, no card needed), create a Valkey service, pick a region.
2. Copy the **Service URI** from the Aiven console — looks like
   `rediss://default:<password>@<host>.aivencloud.com:<port>`.
3. Set it in `.env`:
   ```bash
   VALKEY_URI=rediss://default:<password>@<host>.aivencloud.com:<port>
   ```
   That's it — the `rediss://` scheme auto-enables TLS. For strict
   certificate verification (recommended before real traffic), download
   the CA cert from the Aiven console and also set `VALKEY_CA_CERT=/path/to/ca.pem`.

If Valkey is unreachable under either option, the cache logs a warning and
no-ops (misses only, never errors) — the rest of the app keeps working,
just without the cache-hit cost savings.

Qdrant runs embedded (no separate server) by default — nothing else to install.

## 2. Run the full stack

Two processes, in separate terminals:

```bash
# terminal 1 — backend
export PYTHONPATH=.
uvicorn app:app --reload --port 8000

# terminal 2 — frontend
export API_BASE=http://127.0.0.1:8000
streamlit run frontend/streamlit_app.py
```

Open the Streamlit URL it prints (default `http://localhost:8501`). Register
an account, pick a role, and go — the sidebar and chat surface only what
that role is authorized to see.

## 3. Authentication & authorization

JWT-based (`auth/security.py`, PyJWT + PBKDF2 password hashing — no compiled
dependency risk), user records in SQLite (`auth/store.py`), role checks as
FastAPI dependencies (`auth/deps.py`) so every protected route declares its
own requirement instead of trusting a body field:

```python
@app.post("/ingest")
def ingest(req: IngestRequest, user: CurrentUser = Depends(require_role("compliance_officer", "admin"))):
    ...
```

**Roles**: `claims_adjuster`, `compliance_officer`, `relationship_manager`,
`admin`. Registration is self-serve with role selection for this demo —
production should switch to admin-provisioned accounts (the schema and
`create_user()` already support it; only the `/auth/register` route's open
self-selection needs restricting).

**What's gated**:
- `/ingest` — `compliance_officer` or `admin` only
- `/admin/stats` — `admin` only
- `/query`, `/sessions*`, `/feedback` — any authenticated user; the
  document/chunk access itself is further scoped by the caller's role via
  the retrieval-layer `access_role` filter (a `claims_adjuster` and a
  `relationship_manager` asking the same question can retrieve different
  chunks)
- Session ownership is enforced at the SQL level (`sessions/store.py`), not
  just in the route handler — a session_id from another user returns empty,
  never another user's data, even if guessed correctly

## 4. Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| POST | `/auth/register` | none | create a user |
| POST | `/auth/login` | none | OAuth2 password flow → JWT |
| GET | `/auth/me` | any | current identity |
| POST | `/sessions` | any | create a chat session |
| GET | `/sessions` | any | list caller's own sessions |
| GET | `/sessions/{id}/messages` | any (own) | full transcript |
| DELETE | `/sessions/{id}` | any (own) | delete a session |
| POST | `/query` | any | the RAG pipeline |
| POST | `/ingest` | compliance_officer, admin | document ingestion |
| POST | `/feedback` | any | thumbs up/down, feeds the eval loop |
| GET | `/admin/stats` | admin | trace completeness, refusal rate |
| GET | `/health` | none | liveness |

## 5. What `/query` actually does

Nine stages, every one traced (Langfuse or local JSONL):

```
input guardrail (LLM Guard) → cache check (Valkey, user-scoped) →
memory fetch (Mem0, bias only) → hybrid retrieval (Qdrant) →
rerank (Cohere) → context guardrail (injection re-scan, access re-check,
staleness check, relevance floor, PII mask) → generation
(LiteLLM: Groq → Gemini) → output guardrail (Guardrails AI schema +
groundedness) → cache/memory/session write-back
```

A refused or fail-closed query short-circuits early but still writes to the
session transcript and the trace — nothing about a "no" is invisible.

## 6. What's real, what falls back

| Component | Real path (needs) | Fallback |
|---|---|---|
| Vector DB | Qdrant, embedded — always real | — |
| Injection scan | LLM Guard ML scanner (needs HuggingFace hub) | Regex check |
| Reranker | Cohere Rerank 3 (`COHERE_API_KEY`) | Lexical scorer |
| Generation | Groq/Gemini via LiteLLM | Offline mock |
| Memory | Mem0 (`GEMINI_API_KEY`/`GROQ_API_KEY`) | JSON store |
| Cache | Valkey — always real if server running | No-ops (misses only) |
| Observability | Langfuse (`LANGFUSE_*` keys) | Local JSONL (always dual-written) |
| Eval judge | Groq/Gemini | Lexical-overlap proxy |
| RAGAS | — | Not installed (ADR 0008, hard dependency conflict); DeepEval + custom metrics instead |

`pytest eval/test_golden_set.py` shows Precision@5 = 0.714 against a 0.80
target — correctly failing, because that target assumes real Cohere
reranking and no key is set here. That's the CI gate working as designed.

## 7. Two real bugs this build caught

1. **Access-control no-op** (`ingestion/store.py`): the original filter
   logic treated `"*"` appearing anywhere in the query's allowed-values list
   as an instruction to skip filtering entirely — since every role query
   always included `"*"` (to also match public chunks), role filtering
   never actually applied. Fixed; verified with a live test showing an
   `underwriting_only` role now correctly sees only wildcard-tagged chunks.
2. **Qdrant embedded-mode single-client lock** (`ingestion/run.py`): calling
   `run_ingestion()` from `/ingest` opened a second `VectorStore()` while
   the app's module-level instance already held the storage path open,
   raising a runtime error. Fixed by threading the app's existing store
   instances through as optional parameters instead of always opening a
   fresh connection — the CLI entrypoint still opens its own when run
   standalone.

## 8. Project layout

```
app.py               FastAPI harness — all endpoints, the 9-stage /query pipeline
frontend/
  streamlit_app.py    login/register, chat, ingest (role-gated), admin dashboard
auth/                 JWT + PBKDF2 password hashing, SQLite user store, role deps
sessions/             chat session + message store, ownership enforced in SQL
ingestion/            safety_scan, parser, chunker, versioning, store (Qdrant), run
retrieval/            embed, index (hybrid+RRF), rerank (Cohere), cache (Valkey),
                       query, llamaindex_retriever (LlamaIndex BaseRetriever wrapper)
guardrails/           input_guardrail (LLM Guard), context_guardrail (5 checks),
                       output_guardrail (Guardrails AI), tests/
memory/                store (Mem0 + JSON fallback), smoke_test
gateway/               llm_gateway (LiteLLM: Groq → Gemini, circuit breaker)
eval/                  test_golden_set (DeepEval CI gate), ragas_metrics, golden_set/
cache/                 smoke_test
observability/         tracer (Langfuse + JSONL), trace_check
samples/                synthetic test documents (not real regulatory text)
data/                   local state — Qdrant collections, SQLite, traces. gitignore this.
docs_framework_comparison.md   supplementary per-layer framework trade-off detail
```

## 9. Going further

1. Add `COHERE_API_KEY`, `GROQ_API_KEY`, `GEMINI_API_KEY` to clear the
   precision target and get real generation instead of the offline mock.
2. Restrict `/auth/register`'s open role selection to admin-provisioned
   accounts before any real deployment — the demo leaves it open so the
   Streamlit registration flow is testable end to end without a seed script.
3. Tighten `CORSMiddleware`'s `allow_origins=["*"]` to the actual Streamlit
   origin once deployed somewhere with a fixed domain.
4. Point `VALKEY_HOST`/`QDRANT_URL` at real managed or self-hosted instances
   for multi-process/HA deployment.
5. Expand `eval/golden_set/sample.jsonl` with real Q&A pairs.
6. All 8 milestones in `.genesis/PLAN.md` plus this auth/frontend layer are
   built and individually verified — the genesis kit's L4 VERIFY loop
   (separate agent/model session) is the right next step to check this
   against `.genesis/context-graph.json`'s invariants before calling any
   milestone truly done.
