"""
MedRag - Streamlit Web Application
Adaptive Evidence-Aware Retrieval and Verification Framework for Medical QA
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import List, Optional

import streamlit as st

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from config import MedRagConfig
from main import _build_sample_corpus, load_corpus_file
from medrag import Action, Document, LLMClient, MedRagPipeline

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedRag — Medical Evidence Verification",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Global styling */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a73e8;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #5f6368;
        margin-bottom: 1.5rem;
    }
    
    /* Status Badges */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 14px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        margin-right: 6px;
    }
    .badge-stop {
        background-color: #e6f4ea;
        color: #137333;
        border: 1px solid #ceead6;
    }
    .badge-abstain {
        background-color: #fce8e6;
        color: #c5221f;
        border: 1px solid #fad2cf;
    }
    .badge-supported {
        background-color: #e6f4ea;
        color: #137333;
    }
    .badge-conflicted {
        background-color: #fce8e6;
        color: #c5221f;
    }
    .badge-missing {
        background-color: #fef7e0;
        color: #b06000;
    }
    .badge-uncertain {
        background-color: #f3e8fd;
        color: #7627bb;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #70757a;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Claim card */
    .claim-box {
        background: #fdfdfd;
        border-left: 4px solid #1a73e8;
        border-radius: 4px;
        padding: 12px 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Pipeline Caching & Session State ──────────────────────────────────────────
@st.cache_resource
def get_sample_corpus():
    return _build_sample_corpus()


def get_active_config() -> MedRagConfig:
    return MedRagConfig.from_env()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://api.iconify.design/material-symbols:health-and-safety-outline.svg?color=%231a73e8",
        width=48,
    )
    st.title("MedRag Settings")

    # Load defaults from .env
    env_cfg = get_active_config()

    st.subheader("1. LLM Backend")
    backend_options = ["groq", "openai", "ollama", "mock"]
    default_backend_idx = (
        backend_options.index(env_cfg.llm_backend)
        if env_cfg.llm_backend in backend_options
        else 0
    )
    backend = st.selectbox("Backend Provider", backend_options, index=default_backend_idx)

    # Model input
    default_model = env_cfg.llm_model or ("openai/gpt-oss-20b" if backend == "groq" else "gpt-4o-mini")
    model_name = st.text_input("Model Name (from .env)", value=default_model)

    api_key = env_cfg.llm_api_key or ""
    if backend == "groq":
        st.caption("✅ Groq API Key loaded from `.env`" if api_key else "⚠️ No Groq API Key found in `.env`")
    elif backend == "openai":
        st.caption("✅ OpenAI API Key loaded from `.env`" if api_key else "⚠️ No OpenAI API Key found in `.env`")

    st.divider()

    st.subheader("2. Evidence Verification Thresholds")
    support_threshold = st.slider(
        "Min Support (Entailment)",
        min_value=0.50,
        max_value=0.95,
        value=float(env_cfg.support_threshold),
        step=0.05,
        help="Minimum NLI entailment score required to mark a claim as Supported",
    )
    conflict_threshold = st.slider(
        "Max Conflict (Contradiction)",
        min_value=0.10,
        max_value=0.60,
        value=float(env_cfg.conflict_threshold),
        step=0.05,
        help="Contradiction score above which a claim is considered Conflicted",
    )
    max_rounds = st.slider(
        "Max Retrieval Rounds",
        min_value=1,
        max_value=5,
        value=int(env_cfg.max_search_rounds),
        help="Maximum adaptive search cycles for missing evidence",
    )
    use_reranker = st.checkbox(
        "Cross-Encoder Reranker",
        value=bool(env_cfg.use_reranker),
        help="Re-rank top BM25 + Dense candidates using ms-marco cross-encoder",
    )

    st.divider()

    st.subheader("3. Knowledge Base")
    corpus_choice = st.radio("Corpus Source", ["Built-in Clinical Corpus (15 docs)", "Upload Custom JSON/JSONL"])
    uploaded_file = None
    if corpus_choice == "Upload Custom JSON/JSONL":
        uploaded_file = st.file_uploader("Upload corpus file", type=["json", "jsonl"])

# ── Main Header ───────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🩺 MedRag</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Adaptive Evidence-Aware Retrieval and Clinical Verification Framework for Reliable Medical QA</div>',
    unsafe_allow_html=True,
)

# ── Quick Example Questions ───────────────────────────────────────────────────
st.markdown("**Example Clinical Questions:**")
examples = [
    "Is metformin safe for patients with renal impairment?",
    "Should aspirin be used for primary prevention of cardiovascular events in elderly patients?",
    "What anticoagulant is preferred for stroke prevention in non-valvular atrial fibrillation?",
    "Can SGLT2 inhibitors be prescribed in patients with chronic kidney disease?",
]

cols = st.columns(len(examples))
selected_example = None
for i, ex in enumerate(examples):
    short_label = ex if len(ex) < 45 else ex[:42] + "..."
    if cols[i].button(short_label, key=f"ex_{i}", use_container_width=True):
        st.session_state["query_input"] = ex

# Query Input
query_val = st.session_state.get("query_input", "")
user_question = st.text_input(
    "Enter clinical question:",
    value=query_val,
    placeholder="e.g. Is metformin safe for patients with renal impairment?",
    key="main_query_input",
)

col_btn, col_info = st.columns([1, 4])
run_clicked = col_btn.button("🚀 Analyze & Verify", type="primary", use_container_width=True)

# ── Execution Logic ───────────────────────────────────────────────────────────
if run_clicked:
    if not user_question.strip():
        st.warning("Please enter or select a medical question to analyze.")
        st.stop()

    # Determine corpus
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".jsonl"):
                content = uploaded_file.getvalue().decode("utf-8")
                items = [json.loads(line) for line in content.splitlines() if line.strip()]
            else:
                items = json.loads(uploaded_file.getvalue().decode("utf-8"))
            if not isinstance(items, list):
                items = [items]
            documents = [
                Document(
                    doc_id=str(d.get("id", i)),
                    title=d.get("title", ""),
                    text=d.get("text", d.get("abstract", "")),
                    source=d.get("source", ""),
                    year=int(d.get("year", 2024)),
                    quality_score=float(d.get("quality_score", 0.5)),
                )
                for i, d in enumerate(items)
            ]
            st.success(f"Loaded custom corpus with {len(documents)} documents.")
        except Exception as e:
            st.error(f"Error parsing uploaded file: {e}")
            st.stop()
    else:
        documents = get_sample_corpus()

    # Build config & LLM Client
    cfg = MedRagConfig.from_env()
    cfg.llm_backend = backend
    cfg.llm_model = model_name
    cfg.support_threshold = support_threshold
    cfg.conflict_threshold = conflict_threshold
    cfg.max_search_rounds = max_rounds
    cfg.use_reranker = use_reranker

    llm = LLMClient(
        backend=cfg.llm_backend,
        model=cfg.llm_model,
        api_key=cfg.llm_api_key,
        base_url=cfg.llm_base_url,
    )

    with st.spinner("Running 6-phase adaptive retrieval, claim decomposition, and NLI verification..."):
        t_start = time.time()
        pipeline = MedRagPipeline(documents=documents, llm_client=llm, config=cfg)
        result = pipeline.answer(user_question.strip())
        latency = time.time() - t_start

    st.session_state["latest_result"] = result
    st.session_state["latest_latency"] = latency

# ── Render Results ────────────────────────────────────────────────────────────
if "latest_result" in st.session_state:
    result = st.session_state["latest_result"]
    latency = st.session_state.get("latest_latency", 0.0)
    state = result.evidence_state
    summary = state.summary_dict() if state else {}

    st.markdown("---")
    st.subheader("Clinical Reasoning & Verification Report")

    # Metrics Summary Row
    m1, m2, m3, m4, m5, m6 = st.columns(6)

    # Decision Badge
    is_stop = result.action == Action.STOP
    decision_color = "#137333" if is_stop else "#c5221f"
    decision_bg = "#e6f4ea" if is_stop else "#fce8e6"
    decision_text = "VERIFIED (STOP)" if is_stop else "ABSTAINED (RISK/GAP)"

    with m1:
        st.markdown(
            f"""
            <div class="metric-card" style="border-color:{decision_color}; background-color:{decision_bg};">
                <div class="metric-label">Decision</div>
                <div class="metric-val" style="color:{decision_color}; font-size:1.1rem;">{decision_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Rounds</div>
                <div class="metric-val">{result.rounds}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m3:
        sup_score = summary.get("mean_support", 0.0)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Mean Support</div>
                <div class="metric-val" style="color:#137333;">{sup_score:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m4:
        con_score = summary.get("max_conflict", 0.0)
        con_color = "#c5221f" if con_score >= conflict_threshold else "#5f6368"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Max Conflict</div>
                <div class="metric-val" style="color:{con_color};">{con_score:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m5:
        risk_score = summary.get("clinical_risk", 0.0)
        risk_color = "#c5221f" if risk_score >= 0.7 else "#137333"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Clinical Risk</div>
                <div class="metric-val" style="color:{risk_color};">{risk_score:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m6:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Latency</div>
                <div class="metric-val">{latency:.1f}s</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # Tabbed Interface
    tab_ans, tab_claims, tab_trajectory, tab_sources = st.tabs(
        [
            "💬 Clinical Answer",
            "🔬 Claim Verification Matrix",
            "🔄 Adaptive Rounds (Trajectory)",
            "📚 Consulted Evidence Sources",
        ]
    )

    # ── Tab 1: Answer ─────────────────────────────────────────────────────────
    with tab_ans:
        if result.action == Action.ABSTAIN:
            st.error(
                "⚠️ **MedRag Abstained from generating a full clinical recommendation.**\n\n"
                "The adaptive controller determined that the evidence in the corpus is either conflicting, "
                "lacking, or does not meet the strict safety threshold required for this clinical risk profile."
            )
        else:
            st.success("✅ **Verified Clinical Statement:** All claims are confirmed by retrieved literature.")

        st.markdown(result.final_answer)

    # ── Tab 2: Claims Matrix ──────────────────────────────────────────────────
    with tab_claims:
        if state and state.claims:
            st.markdown(f"**Decomposed into {len(state.claims)} Atomic Clinical Claims:**")

            badge_map = {
                "supported": ("badge-supported", "✓ SUPPORTED"),
                "conflicted": ("badge-conflicted", "✗ CONFLICTED"),
                "missing": ("badge-missing", "? MISSING EVIDENCE"),
                "uncertain": ("badge-uncertain", "~ UNCERTAIN"),
            }

            for idx, c in enumerate(state.claims):
                badge_class, badge_label = badge_map.get(
                    c.status.value, ("badge-uncertain", c.status.value.upper())
                )

                with st.expander(f"Claim {idx+1}: {c.claim}", expanded=(c.status.value != "supported")):
                    st.markdown(f'<span class="badge {badge_class}">{badge_label}</span>', unsafe_allow_html=True)
                    st.write("")

                    c1, c2 = st.columns(2)
                    c1.progress(
                        min(max(c.support_score, 0.0), 1.0),
                        text=f"Entailment / Support: {c.support_score:.2%}",
                    )
                    c2.progress(
                        min(max(c.conflict_score, 0.0), 1.0),
                        text=f"Contradiction / Conflict: {c.conflict_score:.2%}",
                    )

                    if c.best_doc:
                        st.markdown(
                            f"**Top Evidence Match:** *{c.best_doc.title}* ({c.best_doc.source}, {c.best_doc.year})"
                        )
                        st.caption(f"Excerpt: \"{c.best_doc.text[:350]}...\"")
        else:
            st.info("No decomposed claims available for this query.")

    # ── Tab 3: Trajectory ─────────────────────────────────────────────────────
    with tab_trajectory:
        st.markdown("**Adaptive Retrieval & Verification Log:**")
        if result.trajectory:
            for step in result.trajectory:
                rnd = step.get("round", 0) + 1
                q = step.get("query", "")
                docs_count = step.get("new_docs", 0)
                claims_count = step.get("n_claims", 0)
                act = step.get("action", "")
                reason = step.get("reason", "")

                with st.container():
                    st.markdown(f"#### Round {rnd}")
                    st.markdown(f"- **Search Query:** `{q}`")
                    st.markdown(f"- **Retrieved:** {docs_count} new documents | **Claims Evaluated:** {claims_count}")
                    st.markdown(f"- **Controller Decision:** `{act}`")
                    st.caption(f"Reason: {reason}")
                    st.divider()
        else:
            st.info("Single-round retrieval completed.")

    # ── Tab 4: Sources ────────────────────────────────────────────────────────
    with tab_sources:
        st.markdown(f"**Total Consulted Documents: {len(result.retrieved_docs)}**")
        for i, rdoc in enumerate(result.retrieved_docs):
            doc = rdoc.document
            with st.expander(f"[{i+1}] {doc.title} ({doc.year} · {doc.source}) — Quality: {doc.quality_score:.2f}"):
                st.write(doc.text)
                st.caption(f"Doc ID: {doc.doc_id} | Final Relevance Score: {rdoc.final_score:.4f}")
