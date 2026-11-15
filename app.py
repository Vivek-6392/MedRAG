"""
MedRag - Streamlit Web Application
Adaptive Evidence-Aware Retrieval and Verification Framework for Medical QA
"""
from __future__ import annotations

import json
import os
import sys
import time

import streamlit as st

# Prevent OpenMP runtime clash on Windows
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Prevent Streamlit watcher crash on torch.classes inspection
try:
    import torch
    if hasattr(torch, "classes") and not hasattr(torch.classes, "__path__"):
        torch.classes.__path__ = []
except Exception:
    pass

from config import DEFAULT_GROQ_MODEL, MedRagConfig
from main import _build_sample_corpus
from medrag import LLMClient, MedRagPipeline

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedRag — Medical Evidence Verification",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
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
        color: #000000;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #70759a;
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


@st.cache_resource
def get_pubmedqa_corpus(data_path: str = "ori_pqal.json", limit: int = 1000):
    """Loads peer-reviewed PubMed Central research abstract passages from ori_pqal.json."""
    from medrag.benchmarks.adapters.pubmedqa import PubMedQAAdapter
    if not os.path.exists(data_path):
        return []
    adapter = PubMedQAAdapter(data_path=data_path)
    docs = adapter.get_default_corpus()
    return docs[:limit] if docs else []


@st.cache_resource
def get_medqa_textbook_corpus(book_filename: str = "First_Aid_Step1.txt", max_chunks: int = 500):
    """Extracts, cleans, and chunks official MedQA medical textbooks with provenance metadata."""
    from medrag.benchmarks.corpus_preparer import TextbookCorpusPreparer
    p = TextbookCorpusPreparer(chunk_size=800, chunk_overlap=150)
    path = os.path.join("data_clean", "textbooks", "en", book_filename)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    title = book_filename.replace(".txt", "").replace("_", " ")
    return p.chunk_textbook(text, book_title=title, source_filename=book_filename)[:max_chunks]


from medrag.document_validator import parse_uploaded_medical_document


def get_active_config() -> MedRagConfig:
    return MedRagConfig.from_env()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
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
    if backend == env_cfg.llm_backend and env_cfg.llm_model:
        default_model = env_cfg.llm_model
    elif backend == "groq":
        default_model = DEFAULT_GROQ_MODEL
    elif backend == "openai":
        default_model = "gpt-4o-mini"
    else:
        default_model = ""
    model_name = st.text_input("Model Name (from .env)", value=default_model)

    has_groq_key = bool(os.environ.get("GROQ_API_KEY"))
    has_openai_key = bool(os.environ.get("OPENAI_API_KEY"))
    if backend == "groq":
        st.caption("✅ Groq API Key detected in environment" if has_groq_key else "⚠️ No Groq API Key found in `.env`")
    elif backend == "openai":
        st.caption("✅ OpenAI API Key detected in environment" if has_openai_key else "⚠️ No OpenAI API Key found in `.env`")
    elif backend == "ollama":
        st.caption("ℹ️ Ollama uses local server (no API key required)")
    else:
        st.caption("ℹ️ Mock backend uses offline rule-based extraction (for testing)")

    st.divider()

    st.subheader("2. Evidence Verification Thresholds")
    support_threshold = st.slider(
        "Min Support (Entailment)",
        min_value=0.25,
        max_value=0.95,
        value=float(env_cfg.support_threshold),
        step=0.05,
        help="Minimum NLI entailment score required to mark a claim as Supported",
    )
    conflict_threshold = st.slider(
        "Max Conflict (Contradiction)",
        min_value=0.10,
        max_value=0.95,
        value=float(env_cfg.conflict_threshold),
        step=0.05,
        help="Contradiction score above which a claim is considered Conflicted",
    )
    clinical_confidence_threshold = st.slider(
        "Min Clinical Confidence",
        min_value=0.40,
        max_value=0.95,
        value=float(env_cfg.clinical_confidence_threshold),
        step=0.05,
        help="Minimum heuristic confidence after source quality and recency are considered",
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
    vector_store_choice = st.selectbox(
        "Vector Database Engine",
        options=["ChromaDB (Persistent on Disk)", "FAISS (High-Performance In-Memory)", "NumPy (Standard)"],
        index=0,
        help="Vector database engine storing dense embeddings with cosine similarity search.",
    )
    vector_store_backend = "chromadb" if "ChromaDB" in vector_store_choice else ("faiss" if "FAISS" in vector_store_choice else "numpy")
    if vector_store_backend == "chromadb":
        st.caption("⚡ Persistent storage active at `./data/chroma_db`")

    st.markdown("**NLI Verification Engine**")
    nli_options = [
        "ygivenx/modernbert-base-mednli (Clinical ModernBERT on MedNLI)",
        "cross-encoder/nli-MiniLM2-L6-H768 (Default Fast Cross-Encoder)",
        "pritamdeka/PubMedBERT-MNLI-MedNLI (PubMedBERT MedNLI)",
        "Custom Model",
    ]
    default_nli_idx = 0 if "modernbert" in env_cfg.nli_model.lower() else (2 if "pubmedbert" in env_cfg.nli_model.lower() else 1)
    nli_choice = st.selectbox(
        "NLI Verification Model",
        options=nli_options,
        index=default_nli_idx,
        help="Model used for sentence-level clinical claim verification and contradiction detection.",
    )
    if "modernbert" in nli_choice.lower() or "ygivenx" in nli_choice.lower():
        selected_nli_model = "ygivenx/modernbert-base-mednli"
    elif "pubmedbert" in nli_choice.lower():
        selected_nli_model = "pritamdeka/PubMedBERT-MNLI-MedNLI"
    elif "minilm" in nli_choice.lower() or "cross-encoder" in nli_choice.lower():
        selected_nli_model = "cross-encoder/nli-MiniLM2-L6-H768"
    elif "custom" in nli_choice.lower():
        selected_nli_model = st.text_input("Custom NLI Model Name", value=env_cfg.nli_model)
    else:
        selected_nli_model = "ygivenx/modernbert-base-mednli"

    if "ygivenx" in selected_nli_model:
        has_token = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
        st.caption("🔒 *ygivenx/modernbert-base-mednli* is a gated MedNLI model requiring Hugging Face authorization." if not has_token else "✅ Hugging Face authentication detected")
        hf_token_input = st.text_input(
            "Hugging Face Token (HF_TOKEN)",
            value=os.environ.get("HF_TOKEN", ""),
            type="password",
            help="User access token with permission to access ygivenx/modernbert-base-mednli on Hugging Face.",
        )
        if hf_token_input:
            os.environ["HF_TOKEN"] = hf_token_input.strip()

    st.divider()

    st.subheader("3. Knowledge Base")
    has_pubmed = os.path.exists("ori_pqal.json")
    has_medqa = os.path.exists(os.path.join("data_clean", "textbooks", "en"))

    corpus_options = []
    if has_pubmed:
        corpus_options.append("PubMedQA Real Research Abstracts (ori_pqal.json)")
    if has_medqa:
        corpus_options.append("MedQA Textbook: First Aid Step 1")
        corpus_options.append("MedQA Textbook: Pathoma Pathology")
        corpus_options.append("MedQA Textbook: Pharmacology (Katzung)")
    corpus_options.append("📄 Upload Medical Document (PDF, DOCX, TXT)")
    corpus_options.append("Built-in Sample Corpus (15 docs)")
    corpus_options.append("Upload Custom JSON/JSONL")

    corpus_choice = st.radio("Corpus Source", corpus_options, index=0)
    uploaded_file = None
    if corpus_choice == "📄 Upload Medical Document (PDF, DOCX, TXT)":
        uploaded_file = st.file_uploader(
            "Upload clinical guideline, paper, or report",
            type=["pdf", "docx", "doc", "txt", "md"],
            help="Upload a PDF, Word document, or text file to query and verify with MedRAG.",
        )
        if uploaded_file is not None:
            st.caption(f"📎 **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB ready)")
    elif corpus_choice == "Upload Custom JSON/JSONL":
        uploaded_file = st.file_uploader("Upload corpus file", type=["json", "jsonl"])

    # Resolve active knowledge base documents and collection name immediately upon selection
    active_docs = []
    active_collection = "medrag_corpus"

    if corpus_choice == "PubMedQA Real Research Abstracts (ori_pqal.json)":
        active_collection = "medrag_pubmedqa"
        active_docs = get_pubmedqa_corpus()
    elif corpus_choice == "MedQA Textbook: First Aid Step 1":
        active_collection = "medrag_first_aid_step1"
        active_docs = get_medqa_textbook_corpus("First_Aid_Step1.txt")
    elif corpus_choice == "MedQA Textbook: Pathoma Pathology":
        active_collection = "medrag_pathoma"
        active_docs = get_medqa_textbook_corpus("Pathoma_Husain.txt")
    elif corpus_choice == "MedQA Textbook: Pharmacology (Katzung)":
        active_collection = "medrag_pharmacology"
        active_docs = get_medqa_textbook_corpus("Pharmacology_Katzung.txt")
    elif corpus_choice == "Built-in Sample Corpus (15 docs)":
        active_collection = "medrag_sample_corpus"
        active_docs = get_sample_corpus()
    elif corpus_choice == "📄 Upload Medical Document (PDF, DOCX, TXT)" and uploaded_file is not None:
        slug = re.sub(r"[^a-zA-Z0-9_]+", "_", uploaded_file.name).strip("_").lower()[:35]
        active_collection = f"medrag_user_{slug}"
        try:
            active_docs = parse_uploaded_medical_document(uploaded_file.name, uploaded_file.getvalue())
        except Exception as e:
            st.error(f"Error parsing document: {e}")
            active_docs = []
    elif corpus_choice == "Upload Custom JSON/JSONL" and uploaded_file is not None:
        from medrag.document_validator import validate_corpus
        slug = re.sub(r"[^a-zA-Z0-9_]+", "_", uploaded_file.name).strip("_").lower()[:35]
        active_collection = f"medrag_custom_{slug}"
        try:
            if uploaded_file.name.endswith(".jsonl"):
                content = uploaded_file.getvalue().decode("utf-8", errors="replace")
                items = [json.loads(line) for line in content.splitlines() if line.strip()]
            else:
                items = json.loads(uploaded_file.getvalue().decode("utf-8", errors="replace"))
            if not isinstance(items, list):
                items = [items]
            active_docs = validate_corpus(items)
        except Exception as e:
            st.error(f"Error validating corpus: {e}")
            active_docs = []

    # Vector DB Indexing Status & Auto-Index immediately upon selection
    if active_docs and vector_store_backend == "chromadb":
        from medrag.vector_store import ChromaVectorStore
        from medrag.retriever import DenseRetriever

        store = ChromaVectorStore(collection_name=active_collection, persist_dir="./data/chroma_db")
        is_already_indexed = store.count() >= len(active_docs) and store.has_ids([active_docs[0].doc_id, active_docs[-1].doc_id])

        if is_already_indexed:
            st.success(f"🟢 **Vector DB Ready:** {store.count():,} passages cached in `{active_collection}`", icon="💾")
        else:
            with st.spinner(f"Indexing {len(active_docs):,} passages into ChromaDB (`{active_collection}`)..."):
                dense = DenseRetriever(vector_backend="chromadb", collection_name=active_collection, persist_dir="./data/chroma_db")
                dense.index(active_docs)
            st.success(f"✅ **Indexed in ChromaDB:** {len(active_docs):,} passages saved to `{active_collection}`", icon="⚡")
    elif active_docs:
        st.info(f"📚 {len(active_docs):,} passages loaded for search.", icon="ℹ️")

# ── Main Header ───────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🩺 MedRag</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Adaptive Evidence-Aware Retrieval and Clinical Verification Framework for Reliable Medical QA</div>',
    unsafe_allow_html=True,
)

# ── Quick Example & Benchmark Questions ───────────────────────────────────────
st.markdown("**Sample Inquiries & Real Benchmark Questions:**")
examples = [
    ("Metformin and kidney function", "Is metformin safe for patients with renal impairment?"),
    (
        "Aspirin for older adults",
        "Should aspirin be used for primary prevention of cardiovascular events in elderly patients?",
    ),
    (
        "Anticoagulants for atrial fibrillation",
        "What anticoagulant is preferred for stroke prevention in non-valvular atrial fibrillation?",
    ),
    (
        "SGLT2 inhibitors and kidney disease",
        "Can SGLT2 inhibitors be prescribed in patients with chronic kidney disease?",
    ),
]
tab_practice, tab_pubmedqa, tab_medqa = st.tabs([
    "🩺 Practice Cases",
    "🔬 Real PubMedQA Research Inquiries",
    "📚 Real MedQA USMLE Board Inquiries",
])

with tab_practice:
    cols = st.columns(2)
    for i, (label, question) in enumerate(examples):
        if cols[i % 2].button(label, key=f"ex_{i}", use_container_width=True):
            st.session_state["main_query_input"] = question

with tab_pubmedqa:
    if has_pubmed:
        pqa_samples = [
            ("Mitochondria in plant cell death (Gold: yes)", "Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?"),
            ("Postoperative cognitive dysfunction (Gold: no)", "Does postoperative cognitive dysfunction predict one-year mortality in elderly patients?"),
            ("Intraoperative MRI in glioma surgery (Gold: yes)", "Does intraoperative MRI improve extent of resection in glioblastoma surgery?"),
            ("Preoperative statins and AKI (Gold: maybe)", "Do preoperative statins prevent acute kidney injury after cardiac surgery?"),
        ]
        p_cols = st.columns(2)
        for i, (label, q_text) in enumerate(pqa_samples):
            if p_cols[i % 2].button(label, key=f"pqa_{i}", use_container_width=True):
                st.session_state["main_query_input"] = q_text
    else:
        st.info("Place ori_pqal.json in the project root to enable PubMedQA inquiries.")

with tab_medqa:
    if has_medqa:
        medqa_samples = [
            (
                "Carpal tunnel median nerve laceration (USMLE)",
                "A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending surgeon. During closure of the skin incision, the resident accidentally lacerates the palmar cutaneous branch of the median nerve. Which sensory deficit will the patient experience?",
            ),
            (
                "Methotrexate toxicity acute rescue (USMLE)",
                "A 45-year-old female with rheumatoid arthritis on chronic methotrexate therapy presents with stomatitis, nausea, and pancytopenia. What agent should be administered immediately to rescue methotrexate toxicity?",
            ),
            (
                "Exertional heat stroke emergency management (USMLE)",
                "A 24-year-old male marathon runner collapses during hot weather. Rectal temperature is 40.8 C (105.4 F). He is delirious and sweating has ceased. What is the most critical immediate intervention?",
            ),
        ]
        for i, (label, q_text) in enumerate(medqa_samples):
            if st.button(label, key=f"medqa_{i}", use_container_width=True):
                st.session_state["main_query_input"] = q_text
    else:
        st.info("Place data_clean in the project root to enable MedQA USMLE inquiries.")

# Active document banner for uploaded medical file mode
if corpus_choice == "📄 Upload Medical Document (PDF, DOCX, TXT)":
    if uploaded_file is not None:
        st.info(f"📄 **Active Medical Document:** `{uploaded_file.name}` ({uploaded_file.size / 1024:.1f} KB). Ask any clinical question below to retrieve and verify evidence directly from this document.")
    else:
        st.warning("📄 **Medical Document Mode:** Please upload a PDF, DOCX, DOC, or TXT file using the sidebar on the left before running a query.")

# Query Input
user_question = st.text_input(
    "Enter clinical question:",
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

    # Verify knowledge base
    documents = active_docs
    if not documents:
        st.error("No documents loaded in the selected Knowledge Base. Please choose or upload a valid corpus in the sidebar.")
        st.stop()

    # Build config & LLM Client with centralized provider resolver
    cfg = MedRagConfig.from_env()
    try:
        cfg.resolve_for_backend(backend=backend, model=model_name or None)
    except Exception as exc:
        st.error(f"Provider configuration error: {exc}")
        st.stop()

    cfg.support_threshold = support_threshold
    cfg.conflict_threshold = conflict_threshold
    cfg.clinical_confidence_threshold = clinical_confidence_threshold
    cfg.max_search_rounds = max_rounds
    cfg.use_reranker = use_reranker
    cfg.vector_store = vector_store_backend
    cfg.collection_name = active_collection
    cfg.nli_model = selected_nli_model

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
    st.session_state.pop("selected_source_id", None)
    st.session_state["llm_error"] = llm.last_error if llm.call_failed else None
    st.session_state["llm_backend"] = cfg.llm_backend
    st.session_state["llm_model"] = cfg.llm_model

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
    action_str = result.action.value if hasattr(result.action, "value") else str(result.action)
    if action_str in ("STOP", "VERIFIED"):
        decision_color = "#137333"
        decision_bg = "#e6f4ea"
        decision_text = "VERIFIED (STOP)"
    elif action_str == "PARTIAL":
        decision_color = "#b06000"
        decision_bg = "#fef7e0"
        decision_text = "PARTIAL (QUALIFIED)"
    elif action_str == "CONFLICTED":
        decision_color = "#c5221f"
        decision_bg = "#fce8e6"
        decision_text = "CONFLICTED"
    else:
        decision_color = "#c5221f"
        decision_bg = "#fce8e6"
        decision_text = "ABSTAINED"

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
        risk_score = summary.get("max_clinical_risk", 0.0)
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
        action_str = result.action.value if hasattr(result.action, "value") else str(result.action)
        if action_str in ("STOP", "VERIFIED"):
            st.success("✅ **Verified Clinical Statement:** All claims are confirmed by retrieved literature without material contradiction.")
        elif action_str == "PARTIAL":
            st.warning("⚠️ **Qualified Partial Clinical Statement:** Some claims are verified by retrieved literature while others remain unconfirmed.")
        elif action_str == "CONFLICTED":
            st.error("⚔️ **Contradictory Clinical Evidence Detected:** The retrieved medical literature contains material disagreements.")
        else:
            st.error(
                "🛑 **MedRag Abstained from generating a clinical recommendation.**\n\n"
                "The adaptive controller determined that the evidence in the corpus is either conflicting, "
                "lacking, or does not meet the strict safety threshold required for this clinical risk profile."
            )
            llm_error = st.session_state.get("llm_error") or ""
            if (
                st.session_state.get("llm_backend") == "groq"
                and "model_not_found" in llm_error.lower()
            ):
                failed_model = st.session_state.get("llm_model", "configured model")
                st.warning(
                    f"Groq rejected model `{failed_model}`. It may have been retired or may not be "
                    "available to this account. Update `GROQ_MODEL` in `.env` or choose an active "
                    "model in the [Groq model catalog](https://console.groq.com/docs/models). "
                    f"Groq's current migration guidance recommends `{DEFAULT_GROQ_MODEL}` as a "
                    "replacement for `llama-3.3-70b-versatile`. Then rerun the query. A successful "
                    "LLM call does not verify the clinical evidence; MedRag may still abstain."
                )

        st.markdown(result.final_answer)

    # ── Tab 2: Claims Matrix ──────────────────────────────────────────────────
    with tab_claims:
        claims_list = (
            getattr(state, "claims", None)
            or getattr(state, "claim_evidences", [])
            if state
            else []
        )
        if claims_list:
            st.markdown(f"**Decomposed into {len(claims_list)} Atomic Clinical Claims:**")

            badge_map = {
                "supported": ("badge-supported", "✓ SUPPORTED"),
                "conflicted": ("badge-conflicted", "✗ CONFLICTED"),
                "missing": ("badge-missing", "? MISSING EVIDENCE"),
                "uncertain": ("badge-uncertain", "~ UNCERTAIN"),
            }

            for idx, c in enumerate(claims_list):
                status_val = c.status.value if hasattr(c.status, "value") else str(c.status)
                badge_class, badge_label = badge_map.get(
                    status_val, ("badge-uncertain", status_val.upper())
                )

                with st.expander(f"Claim {idx+1}: {c.claim}", expanded=(status_val != "supported")):
                    st.markdown(f'<span class="badge {badge_class}">{badge_label}</span>', unsafe_allow_html=True)
                    st.write("")

                    c1, c2, c3 = st.columns(3)
                    c1.progress(
                        min(max(float(c.support_score), 0.0), 1.0),
                        text=f"Support (Entailment): {c.support_score:.1%}",
                    )
                    c2.progress(
                        min(max(float(c.conflict_score), 0.0), 1.0),
                        text=f"Conflict (Contradiction): {c.conflict_score:.1%}",
                    )
                    conf = getattr(c, "clinical_confidence", c.support_score)
                    c3.progress(
                        min(max(float(conf), 0.0), 1.0),
                        text=f"Clinical Confidence: {conf:.1%}",
                    )

                    best_src = getattr(c, "best_source", "")
                    best_snip = getattr(c, "best_snippet", "")
                    rec_yr = getattr(c, "recency_year", "")
                    if best_src:
                        yr_str = f" ({rec_yr})" if rec_yr else ""
                        st.markdown(f"**Top Evidence Match:** *{best_src}*{yr_str}")
                    if best_snip:
                        st.caption(f"Excerpt: \"{best_snip}...\"")

                    # Display individual supporting & contradicting passages if present
                    sup_passages = getattr(c, "supporting_passages", [])
                    con_passages = getattr(c, "contradicting_passages", [])
                    if sup_passages:
                        with st.container(border=True):
                            st.markdown(f"**Supporting Passages ({len(sup_passages)})**")
                            for passage_idx, p in enumerate(sup_passages):
                                st.markdown(
                                    f"- **{p.doc_title}** ({p.year} · {p.source}) · "
                                    f"Quality: `{p.quality_score:.2f}` · "
                                    f"Entailment: `{p.entailment:.1%}` · ID: `{p.doc_id}`"
                                    f"{' · Illustrative sample' if p.illustrative else ''}"
                                )
                                st.caption(p.text)
                                if st.button(
                                    "View full source",
                                    key=f"support_source_{idx}_{passage_idx}",
                                ):
                                    st.session_state["selected_source_id"] = p.doc_id
                    if con_passages:
                        with st.container(border=True):
                            st.markdown(f"**Contradicting Passages ({len(con_passages)})**")
                            for passage_idx, p in enumerate(con_passages):
                                st.markdown(
                                    f"- ⚠️ **{p.doc_title}** ({p.year} · {p.source}) · "
                                    f"Quality: `{p.quality_score:.2f}` · "
                                    f"Contradiction: `{p.contradiction:.1%}` · ID: `{p.doc_id}`"
                                    f"{' · Illustrative sample' if p.illustrative else ''}"
                                )
                                st.caption(p.text)
                                if st.button(
                                    "View full source",
                                    key=f"conflict_source_{idx}_{passage_idx}",
                                ):
                                    st.session_state["selected_source_id"] = p.doc_id
                    neutral_passages = getattr(c, "neutral_passages", [])
                    if neutral_passages:
                        with st.container(border=True):
                            st.markdown(f"**Neutral Passages ({len(neutral_passages)})**")
                            for passage_idx, p in enumerate(neutral_passages):
                                st.markdown(
                                    f"- **{p.doc_title}** ({p.year} · {p.source}) · "
                                    f"Quality: `{p.quality_score:.2f}` · "
                                    f"Neutral: `{p.neutral:.1%}` · ID: `{p.doc_id}`"
                                    f"{' · Illustrative sample' if p.illustrative else ''}"
                                )
                                st.caption(p.text)
                                if st.button(
                                    "View full source",
                                    key=f"neutral_source_{idx}_{passage_idx}",
                                ):
                                    st.session_state["selected_source_id"] = p.doc_id

            selected_source_id = st.session_state.get("selected_source_id")
            if selected_source_id:
                selected_source = next(
                    (
                        rdoc.document
                        for rdoc in result.retrieved_docs
                        if rdoc.document.doc_id == selected_source_id
                    ),
                    None,
                )
                if selected_source:
                    st.markdown("---")
                    st.markdown(f"### Full source: {selected_source.title}")
                    st.caption(
                        f"{selected_source.year} · {selected_source.source} · "
                        f"Quality: {selected_source.quality_score:.2f} · "
                        f"Document ID: {selected_source.doc_id}"
                    )
                    st.write(selected_source.text)
                else:
                    st.warning("The selected source is not available in the retrieved documents.")
        else:
            st.info("No decomposed claims available for this query.")

    # ── Tab 3: Trajectory ─────────────────────────────────────────────────────
    with tab_trajectory:
        st.markdown("**Adaptive Retrieval & Verification Log:**")
        trajectory_log = getattr(result, "trajectory", None) or getattr(result, "history", [])
        if trajectory_log:
            for step in trajectory_log:
                rnd = step.get("round", 1)
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
                final_score = getattr(rdoc, "final_score", getattr(rdoc, "rerank_score", getattr(rdoc, "score", 0.0)))
                st.caption(f"Doc ID: {doc.doc_id} | Final Relevance Score: {final_score:.4f}")

# ── Academic Research & Clinical Safety Disclaimer ────────────────────────────
st.markdown("---")
st.caption(
    "⚠️ **Academic Research Disclaimer:** MedRAG is a research prototype designed to evaluate "
    "retrieval-augmented generation, claim-level evidence verification, and selective abstention. "
    "It is NOT a certified medical device, does NOT guarantee clinical safety or HIPAA compliance, "
    "and must NOT be used for direct clinical diagnosis or patient treatment decisions."
)

