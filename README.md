# BFSI RAG Assistant

FastAPI backend + Streamlit frontend + JWT auth/RBAC, on a full RAG stack:
**LlamaIndex** · **Qdrant** · **Cohere Rerank 3** · **LiteLLM (Groq/Gemini)**
· **LLM Guard + Guardrails AI** (three-layer guardrails) · **Mem0** ·
**Valkey** · **Langfuse** · **DeepEval**.

---

## 1. Install

```bash
cd bfsi-rag-assistant
uv venv
uv sync
cp .env.example .env
```

Requires Python 3.10+ and a running Valkey (or Redis-protocol-compatible)
server.

**Option A — local, self-hosted (default):**
```bash
apt-get install valkey-server valkey-tools
valkey-server --daemonize yes
```

**Option B — Aiven for Valkey (managed, useful if a local install is blocked):**
1. Sign up at [aiven.io](https://aiven.io) (free tier, no card), create a Valkey service, pick a region.
2. Copy the **Service URI** from the Aiven console (`rediss://default:<password>@<host>.aivencloud.com:<port>`).
3. Set it in `.env`:
   ```bash
   VALKEY_URI=rediss://default:<password>@<host>.aivencloud.com:<port>
   ```
   The `rediss://` scheme auto-enables TLS. For strict certificate
   verification, download the CA cert from the Aiven console and also set
   `VALKEY_CA_CERT=/path/to/ca.pem`.

If Valkey is unreachable under either option, the cache logs a warning and
no-ops (misses only, never errors) — the rest of the app keeps working,
just without cache-hit cost savings.

Qdrant runs embedded (no separate server needed) by default.

## 2. Run

Two processes, in separate terminals:

```bash
# terminal 1 — backend
export PYTHONPATH=.
uv run uvicorn app:app --reload --port 8000

# terminal 2 — frontend
export API_BASE=http://127.0.0.1:8000
uv run streamlit run frontend/streamlit_app.py
```

Open the Streamlit URL it prints (default `http://localhost:8501`). Register
an account, pick a role, and go — the sidebar and chat surface only what
that role is authorized to see.

## 3. Authentication & authorization

JWT-based (`auth/security.py` — PyJWT + PBKDF2 password hashing, no compiled
dependency risk), user records in SQLite (`auth/store.py`), role checks as
FastAPI dependencies (`auth/deps.py`) so every protected route declares its
own requirement instead of trusting a body field:

```python
@app.post("/ingest")
def ingest(req: IngestRequest, user: CurrentUser = Depends(require_role("compliance_officer", "admin"))):
    ...
```

**Roles**: `claims_adjuster`, `compliance_officer`, `admin`.

Registration is self-serve with role selection for this demo — production
should switch to admin-provisioned accounts (the schema and `create_user()`
already support it; only `/auth/register`'s open self-selection needs
restricting).

**What's gated**:
- `/ingest` — `compliance_officer` or `admin` only
- `/admin/stats` — `admin` only
- `/query`, `/sessions*`, `/feedback` — any authenticated user; document/chunk
  access is further scoped by the caller's role via the retrieval layer's
  `access_role` filter (a `claims_adjuster`
  asking the same question can retrieve different chunks)
- Session ownership is enforced at the SQL level (`sessions/store.py`), not
  just in the route handler — a session ID from another user returns empty,
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

## 5. What `/query` does

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
| Injection scan | LLM Guard ML scanner (HuggingFace hub) | Regex check |
| Reranker | Cohere Rerank 3 (`COHERE_API_KEY`) | Lexical scorer |
| Generation | Groq/Gemini via LiteLLM | Offline mock |
| Memory | Mem0 (`GEMINI_API_KEY`/`GROQ_API_KEY`) | JSON store |
| Cache | Valkey — always real if server running | No-ops (misses only) |
| Observability | Langfuse (`LANGFUSE_*` keys) | Local JSONL (always dual-written) |
| Eval judge | Groq/Gemini | Lexical-overlap proxy |
| RAGAS | — | Not installed (ADR 0008, hard dependency conflict); DeepEval + custom metrics instead |


## 7. Project layout

```
app.py                FastAPI harness — all endpoints, the 9-stage /query pipeline
frontend/
  streamlit_app.py     login/register, chat, ingest (role-gated), admin dashboard
auth/                  JWT + PBKDF2 password hashing, SQLite user store, role deps
sessions/              chat session + message store, ownership enforced in SQL
ingestion/             safety_scan, parser, chunker, versioning, store (Qdrant), run
retrieval/             embed, index (hybrid+RRF), rerank (Cohere), cache (Valkey),
                        query, llamaindex_retriever (LlamaIndex BaseRetriever wrapper)
guardrails/            input_guardrail (LLM Guard), context_guardrail (5 checks),
                        output_guardrail (Guardrails AI), tests/
memory/                store (Mem0 + JSON fallback), smoke_test
gateway/               llm_gateway (LiteLLM: Groq → Gemini, circuit breaker)
eval/                  test_golden_set (DeepEval CI gate), ragas_metrics, golden_set/
cache/                 smoke_test
observability/         tracer (Langfuse + JSONL), trace_check
samples/               synthetic test documents (not real regulatory text)
data/                  local state — Qdrant collections, SQLite, traces (gitignored)
docs_framework_comparison.md   supplementary per-layer framework trade-off detail
```

## License

MIT