"""
Streamlit frontend for the BFSI RAG Assistant.

Run:  streamlit run frontend/streamlit_app.py

Talks to the FastAPI backend over HTTP — the frontend holds no business
logic, no direct DB/vector-store access, and never calls the LLM gateway
itself. Every guardrail, scoping rule, and fail-closed behavior lives in
the backend; this file is purely presentation + the JWT held in
st.session_state.
"""
from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="BFSI RAG Assistant", page_icon="📋", layout="wide")


# ─────────────────────────── session state ───────────────────────────

def _init_state() -> None:
    defaults = {
        "token": None, "username": None, "role": None, "user_id": None,
        "current_session_id": None, "messages": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


def _auth_headers() -> dict:
    return {"Authorization": f"Bearer {st.session_state.token}"}


def _api(method: str, path: str, timeout: int = 120, **kwargs) -> requests.Response:
    return requests.request(method, f"{API_BASE}{path}", headers=_auth_headers(), timeout=timeout, **kwargs)


def _logout() -> None:
    # 2026-08-24 fix: this used to only clear LOCAL Streamlit state — the
    # token itself stayed fully valid server-side until its natural 24h
    # expiry, meaning the backend's token-revocation feature (item 1) was
    # built but never actually triggered by clicking "Log out" in the UI.
    # Call the real endpoint FIRST, while the token is still in
    # session_state (needed for the Authorization header), then clear.
    if st.session_state.get("token"):
        try:
            _api("POST", "/auth/logout", timeout=10)
        except Exception:
            pass  # logging out locally must succeed even if the backend
            # call fails (e.g. server briefly unreachable) — the token
            # will simply expire naturally in that case, not a security
            # hole, just not immediate revocation
    for k in ("token", "username", "role", "user_id", "current_session_id", "messages"):
        st.session_state[k] = None if k != "messages" else []
    st.rerun()


# ─────────────────────────── auth screens ───────────────────────────

def render_login() -> None:
    st.title("📋 BFSI RAG Assistant")
    st.caption("Compliance & claims policy Q&A — grounded, cited, role-scoped")

    tab_login, tab_register = st.tabs(["Log in", "Register"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", use_container_width=True)
        if submitted:
            r = requests.post(f"{API_BASE}/auth/login", data={"username": username, "password": password}, timeout=15)
            if r.status_code == 200:
                data = r.json()
                st.session_state.token = data["access_token"]
                st.session_state.username = data["username"]
                st.session_state.role = data["role"]
                st.rerun()
            elif r.status_code == 429:
                st.error(r.json().get("detail", "account temporarily locked"))
            else:
                st.error(r.json().get("detail", "login failed"))

    with tab_register:
        st.caption(
            "New accounts start as a claims adjuster. An admin can promote "
            "you to compliance officer or admin afterward."
        )
        with st.form("register_form"):
            username = st.text_input("Username", key="reg_username")
            email = st.text_input("Email")
            password = st.text_input("Password (min 8 chars)", type="password", key="reg_password")
            submitted = st.form_submit_button("Register", use_container_width=True)
        if submitted:
            r = requests.post(
                f"{API_BASE}/auth/register",
                json={"username": username, "email": email, "password": password},
                timeout=15,
            )
            if r.status_code == 201:
                data = r.json()
                st.session_state.token = data["access_token"]
                st.session_state.username = data["username"]
                st.session_state.role = data["role"]
                st.success(f"Registered as {data['role']}. Redirecting…")
                st.rerun()
            else:
                st.error(r.json().get("detail", "registration failed"))


# ─────────────────────────── main app screens ───────────────────────────

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(f"**{st.session_state.username}**")
        st.caption(f"Role: `{st.session_state.role}`")
        if st.button("Log out", use_container_width=True):
            _logout()

        st.divider()
        st.subheader("Chat sessions")
        if st.button("+ New chat", use_container_width=True):
            st.session_state.current_session_id = None
            st.session_state.messages = []
            st.rerun()

        r = _api("GET", "/sessions")
        if r.status_code == 200:
            for s in r.json():
                label = s["title"][:35] or "New chat"
                cols = st.columns([5, 1])
                if cols[0].button(label, key=f"sess_{s['session_id']}", use_container_width=True):
                    st.session_state.current_session_id = s["session_id"]
                    _load_messages(s["session_id"])
                    st.rerun()
                if cols[1].button("🗑", key=f"del_{s['session_id']}"):
                    _api("DELETE", f"/sessions/{s['session_id']}")
                    if st.session_state.current_session_id == s["session_id"]:
                        st.session_state.current_session_id = None
                        st.session_state.messages = []
                    st.rerun()


def _load_messages(session_id: str) -> None:
    r = _api("GET", f"/sessions/{session_id}/messages")
    if r.status_code == 200:
        st.session_state.messages = r.json()


def render_chat() -> None:
    st.title("📋 BFSI compliance & claims assistant")
    st.caption("Answers are grounded in ingested policy/regulatory documents and always cited, or refused.")

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])
            if m.get("citations"):
                with st.expander(f"Sources ({len(m['citations'])})"):
                    for c in m["citations"]:
                        st.caption(f"📄 {c}")
            if m.get("grounded") is False and m["role"] == "assistant":
                st.caption("⚠️ Not grounded in retrieved documents")
            if m.get("degraded") and m["role"] == "assistant":
                st.warning(
                    "⚠️ This answer came from an offline fallback, not a real language "
                    "model — every real provider was unavailable when this was generated. "
                    "Treat it as lower-confidence and verify independently.",
                    icon="⚠️",
                )

    prompt = st.chat_input("Ask about a policy clause, claim process, or regulation…")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt, "citations": [], "grounded": None})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Checking guardrails, retrieving, generating…"):
                r = _api("POST", "/query", json={
                    "query": prompt, "session_id": st.session_state.current_session_id,
                })
            if r.status_code != 200:
                st.error(f"Request failed: {r.status_code} {r.text}")
                return
            data = r.json()
            st.session_state.current_session_id = data["session_id"]
            st.write(data["answer"])

            if data["citations"]:
                with st.expander(f"Sources ({len(data['citations'])})"):
                    for c in data["citations"]:
                        st.caption(f"📄 {c}")
            if not data["grounded"]:
                st.caption("⚠️ Not grounded in retrieved documents")
            if data.get("degraded"):
                st.warning(
                    "⚠️ This answer came from an offline fallback, not a real language "
                    "model — every real provider was unavailable when this was generated. "
                    "Treat it as lower-confidence and verify independently.",
                    icon="⚠️",
                )

            col1, col2, _ = st.columns([1, 1, 8])
            if col1.button("👍", key=f"up_{data['trace_id']}"):
                _api("POST", "/feedback", json={
                    "trace_id": data["trace_id"], "session_id": data["session_id"], "rating": "up",
                })
                st.toast("Thanks for the feedback")
            if col2.button("👎", key=f"down_{data['trace_id']}"):
                _api("POST", "/feedback", json={
                    "trace_id": data["trace_id"], "session_id": data["session_id"], "rating": "down",
                })
                st.toast("Thanks — this gets reviewed")

            st.session_state.messages.append({
                "role": "assistant", "content": data["answer"],
                "citations": data["citations"], "grounded": data["grounded"],
                "degraded": data.get("degraded", False),
            })


def render_ingest() -> None:
    st.title("📥 Document ingestion")
    st.caption("Compliance officer / admin only. Ingests a document already present on the server's filesystem.")

    with st.form("ingest_form"):
        doc_path = st.text_input("Document path", value="samples/sample_health_policy.txt")
        source = st.selectbox("Source type", ["policy_wording", "rbi_irdai_circular"])
        product_line = st.text_input("Product line", value="health")
        access_role = st.multiselect(
            "Access roles (who can retrieve this document's chunks)",
            ["claims_adjuster", "compliance_officer", "admin", "*"],
            default=["claims_adjuster", "compliance_officer"],
        )
        dry_run = st.checkbox("Dry run (compute diff, don't write)", value=False)
        submitted = st.form_submit_button("Ingest", use_container_width=True)

    if submitted:
        with st.spinner("Scanning, chunking, embedding, indexing… large documents can take a minute"):
            r = _api("POST", "/ingest", timeout=180, json={
                "doc_path": doc_path, "source": source, "product_line": product_line,
                "access_role": access_role, "dry_run": dry_run,
            })
        if r.status_code == 200:
            data = r.json()
            if data["status"] == "quarantined":
                st.error(f"Document quarantined: {data['reasons']}")
            else:
                st.success(
                    f"doc_id={data['doc_id']} — new: {data['new_chunks']}, "
                    f"changed: {data['changed_chunks']}, unchanged: {data['unchanged_chunks']}, "
                    f"removed: {data['removed_chunks']}"
                )
                st.json(data.get("sample_metadata", []))
        elif r.status_code == 403:
            st.error("Your role isn't authorized to ingest documents.")
        else:
            st.error(f"Ingestion failed: {r.status_code} {r.text}")


def render_admin() -> None:
    st.title("🛠 Admin dashboard")

    view = st.radio(
        "Dashboard view",
        ["Live traffic", "Evaluation runs", "Blended (both)"],
        horizontal=True,
        help="Live and evaluation traffic are traced separately (item 9) — "
             "an eval run's provider calls, cache checks, and latency used "
             "to silently blend into what looked like live-traffic numbers.",
    )
    request_type_param = {"Live traffic": "query", "Evaluation runs": "eval", "Blended (both)": None}[view]

    r = _api("GET", "/admin/stats", params={"request_type": request_type_param} if request_type_param else {})
    if r.status_code == 403:
        st.error("Admin role required.")
        return
    if r.status_code != 200:
        st.error(f"Failed to load stats: {r.status_code}")
        return
    s = r.json()

    if s["request_count"] == 0:
        st.info(f"No {view.lower()} traced yet.")
        return

    st.caption(f"Computed from the last {s['request_count']} traced requests ({view.lower()}).")

    # the four "glance and know if something's wrong" numbers
    st.subheader("At a glance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("CPSO (cost per grounded answer)", f"${s['cpso']:.5f}" if s["cpso"] is not None else "—")
    col2.metric("Latency p95", f"{s['latency_p95_ms'] * 0.001:.2f}s" if s["latency_p95_ms"] is not None else "—")
    col3.metric("Grounded rate", f"{s['grounded_rate']*100:.0f}%" if s["grounded_rate"] is not None else "—")
    col4.metric("Fail-closed rate", f"{s['fail_closed_rate']*100:.0f}%" if s["fail_closed_rate"] is not None else "—")

    st.divider()
    st.subheader("Cost & tokens")
    col1, col2 = st.columns(2)
    col1.metric("Total estimated cost", f"${s['total_estimated_cost']:.5f}")
    col2.metric("Avg cost / request", f"${s['avg_cost_per_request']:.5f}" if s["avg_cost_per_request"] is not None else "—")

    st.subheader("Latency")
    col1, col2, col3 = st.columns(3)
    col1.metric("p50", f"{s['latency_p50_ms']* 0.001:.0f} s" if s["latency_p50_ms"] is not None else "—")
    col2.metric("p95", f"{s['latency_p95_ms']* 0.001:.0f} s" if s["latency_p95_ms"] is not None else "—")
    col3.metric("p99", f"{s['latency_p99_ms']* 0.001:.0f} s" if s["latency_p99_ms"] is not None else "—")
    if s["stage_latency_p50_ms"]:
        st.caption("Median time elapsed BEFORE each stage started, per request:")

        stage_latency_s = {
            stage: latency_ms / 1000
            for stage, latency_ms in s["stage_latency_p50_ms"].items()
        }

        st.bar_chart(stage_latency_s)

    st.subheader("Guardrails & cache")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cache hit rate", f"{s['cache_hit_rate']*100:.0f}%" if s["cache_hit_rate"] is not None else "—")
    col2.metric("Refusal rate", f"{s['refusal_rate']*100:.0f}%" if s["refusal_rate"] is not None else "—")
    col3.metric("PII masked (chunks)", s["pii_masked_count"])
    col4.metric("Context-guardrail drops", s["context_guardrail_drop_count"])

    if s.get("cache_hit_rate_by_role"):
        st.caption("Cache hit rate by role (scoped since the 2026-08-23 cross-role leak fix):")
        st.bar_chart(s["cache_hit_rate_by_role"])

    col1, col2 = st.columns(2)
    col1.metric(
        "Stale citations", s.get("stale_citation_count", 0),
        help="Should always be 0 — the context guardrail drops superseded chunks before generation. "
             "A nonzero value here means that invariant broke.",
    )
    col2.metric(
        "Feedback ratio (👍/total)",
        f"{s['feedback_ratio']*100:.0f}%" if s.get("feedback_ratio") is not None else "—",
        help=f"{s.get('feedback_up_count', 0)} up, {s.get('feedback_down_count', 0)} down",
    )

    if s.get("provider_reliability"):
        st.caption("Provider reliability (per-attempt outcomes across the LLM gateway):")
        st.dataframe(s["provider_reliability"])

    st.metric("Trace completeness", f"{s['trace_completeness_rate']*100:.0f}%" if s["trace_completeness_rate"] is not None else "—")
    st.caption("Backed by Langfuse when configured, local JSONL trace log otherwise — see observability/tracer.py and observability/dashboard.py.")

    st.divider()
    st.subheader("Evaluation")
    st.caption(
        "Runs the golden-set suite live against the current pipeline. Uses real "
        "API calls if provider keys are configured — a full run makes ~50-75 "
        "real provider calls, easily exceeding free-tier limits (Cohere: "
        "10/min, Gemini: 20/day). Sample a subset to conserve quota during "
        "routine testing."
    )
    sample_size = st.number_input(
        "Retrieval cases to run (of 22 total) — 4 refusal cases always run in full",
        min_value=1, max_value=22, value=5, step=1,
    )
    if st.button("Run eval now"):
        with st.spinner(f"Running golden-set evaluation ({sample_size} of 22 retrieval cases + all refusal cases)…"):
            er = _api("POST", "/admin/eval", timeout=600, params={"sample_size": sample_size})
        if er.status_code == 200:
            ed = er.json()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Recall@25", f"{ed['recall_at_25_doc_level']*100:.0f}%" if ed["recall_at_25_doc_level"] is not None else "—")
            col2.metric("Precision@5", f"{ed['precision_at_5_clause_level']*100:.0f}%" if ed["precision_at_5_clause_level"] is not None else "—")
            col3.metric("MRR", f"{ed['mrr']:.3f}" if ed.get("mrr") is not None else "—")
            col4.metric("nDCG@5", f"{ed['ndcg_at_5']:.3f}" if ed.get("ndcg_at_5") is not None else "—")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Faithfulness", f"{ed['faithfulness_rate']*100:.0f}%")
            col2.metric("Hallucination rate", f"{ed.get('hallucination_rate', 0)*100:.0f}%")
            col3.metric("False refusal rate", f"{ed.get('false_refusal_rate', 0)*100:.0f}%" if ed.get("false_refusal_rate") is not None else "—")
            col4.metric("False pass rate", f"{ed.get('false_pass_rate', 0)*100:.0f}%" if ed.get("false_pass_rate") is not None else "—")

            st.caption(
                f"{ed['grounded_count']} / {ed['n_retrieval_cases']} retrieval cases grounded · "
                f"{ed['n_refusal_cases']} refusal cases checked · refusal accuracy "
                f"{ed['refusal_accuracy']*100:.0f}%" if ed.get("refusal_accuracy") is not None else ""
            )

            if ed.get("falsely_refused_queries"):
                with st.expander(f"⚠️ {len(ed['falsely_refused_queries'])} falsely refused queries — likely a confidence-threshold tuning issue"):
                    st.caption(
                        "These are legitimate, in-domain golden-set questions that got incorrectly "
                        "refused as out-of-domain. If this list is non-empty, check "
                        "OUT_OF_DOMAIN_CONFIDENCE_FLOOR / VAGUE_QUERY_CONFIDENCE_FLOOR in config.py — "
                        "they may need recalibrating against your current embedding model's real "
                        "confidence-score distribution."
                    )
                    for q in ed["falsely_refused_queries"]:
                        st.caption(f"• {q}")
        else:
            st.error(f"Eval run failed: {er.status_code} {er.text}")

    hist = _api("GET", "/admin/eval-history")
    if hist.status_code == 200 and hist.json():
        st.caption("Recall/Precision drift across past eval runs:")
        rows = hist.json()
        chart_data = {
            "recall": [r.get("recall_at_25_doc_level") for r in rows],
            "precision": [r.get("precision_at_5_clause_level") for r in rows],
        }
        st.line_chart(chart_data)

    st.divider()
    st.subheader("Reset dashboard data")
    st.caption(
        "Clears traces, eval history, and feedback for a genuinely fresh "
        "start — e.g. after a heavy testing day when the numbers above no "
        "longer reflect current reality. Archived audit-log files "
        "(data/traces_archive/) are NOT affected — this only clears the "
        "active log."
    )
    confirm_reset = st.checkbox("I understand this clears all dashboard data")
    if st.button("Reset dashboard data", disabled=not confirm_reset):
        r = _api("POST", "/admin/observability/reset", params={"confirm": "true"})
        if r.status_code == 200:
            st.success(f"Cleared: {', '.join(r.json()['cleared'])}")
            st.rerun()
        else:
            st.error(f"Reset failed: {r.status_code} {r.text}")

    st.divider()
    st.subheader("User management")
    ur = _api("GET", "/admin/users")
    if ur.status_code == 200:
        users = ur.json()
        for u in users:
            cols = st.columns([2, 3, 2, 2])
            cols[0].write(u["username"])
            cols[1].caption(u["email"])
            cols[2].caption(u["role"])
            new_role = cols[3].selectbox(
                "role", ["claims_adjuster", "compliance_officer", "admin"],
                index=["claims_adjuster", "compliance_officer", "admin"].index(u["role"]),
                key=f"role_{u['id']}", label_visibility="collapsed",
            )
            if new_role != u["role"]:
                confirm_password = st.text_input(
                    "Confirm your admin password",
                    type="password",
                    key=f"confirm_password_{u['id']}",
                )

                if st.button("Update", key=f"update_{u['id']}"):
                    if not confirm_password:
                        st.warning("Enter your admin password to confirm the role change.")
                    else:
                        pr = _api(
                            "PATCH",
                            f"/admin/users/{u['id']}/role",
                            json={
                                "role": new_role,
                                "confirm_password": confirm_password,
                            },
                        )

                        if pr.status_code == 200:
                            st.success(f"{u['username']} is now {new_role}")
                            st.rerun()
                        elif pr.status_code == 401:
                            st.error("Password confirmation failed. Enter your current admin password.")
                        else:
                            st.error(f"Failed: {pr.status_code} {pr.text}")
    else:
        st.error(f"Failed to load users: {ur.status_code}")


# ─────────────────────────── router ───────────────────────────

if not st.session_state.token:
    render_login()
else:
    render_sidebar()
    tabs = ["💬 Chat"]
    if st.session_state.role in ("compliance_officer", "admin"):
        tabs.append("📥 Ingest")
    if st.session_state.role == "admin":
        tabs.append("🛠 Admin")

    selected = st.tabs(tabs)
    with selected[0]:
        render_chat()
    idx = 1
    if "📥 Ingest" in tabs:
        with selected[idx]:
            render_ingest()
        idx += 1
    if "🛠 Admin" in tabs:
        with selected[idx]:
            render_admin()