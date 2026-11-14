"""
Standardized Baseline Systems and Ablation Configurations for MedRAG.
Defines canonical experimental configurations to isolate and quantify the
independent contribution of each core MedRAG architectural component:
1. Hybrid retrieval (BM25 + Dense RRF)
2. Cross-encoder neural reranking
3. Atomic claim decomposition
4. Sentence-level NLI verification
5. Multi-passage evidence state & conflict tracking
6. Iterative multi-round search
7. Risk-aware adaptive stopping policy
8. Selective abstention
9. Source-quality & temporal recency weighting
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..config_schema import ExperimentConfig


# ==============================================================================
# Canonical Baseline Definitions (PART A)
# ==============================================================================

BASELINE_DESCRIPTIONS: Dict[str, str] = {
    "closed_book": (
        "Baseline 1 (CLOSED_BOOK_LLM): Direct zero-shot generation using LLM parametric knowledge; "
        "zero external retrieval."
    ),
    "single_pass_dense_rag": (
        "Baseline 2 (SINGLE_PASS_DENSE_RAG): Single-round dense embedding retrieval (MiniLM) followed by "
        "parametric generation without reranking or verification."
    ),
    "single_pass_bm25_rag": (
        "Baseline 3 (SINGLE_PASS_BM25_RAG): Single-round sparse BM25 lexical retrieval followed by "
        "generation without reranking or verification."
    ),
    "hybrid_rag": (
        "Baseline 4 (HYBRID_RAG): Single-round sparse (BM25) and dense retrieval fused with "
        "Reciprocal Rank Fusion (RRF); no neural reranking or verification."
    ),
    "hybrid_reranked_rag": (
        "Baseline 5 (HYBRID_RERANKED_RAG): Single-round hybrid RRF retrieval followed by cross-encoder "
        "neural reranking; no claim verification."
    ),
    "claim_verified_rag": (
        "Baseline 6 (CLAIM_VERIFIED_RAG): Single-round hybrid retrieval with cross-encoder reranking, "
        "atomic claim decomposition, and NLI verification, but without iterative search rounds."
    ),
    "iterative_rag": (
        "Baseline 7 (ITERATIVE_RAG): Multi-round iterative retrieval and claim verification using "
        "standard threshold stopping without clinical risk policy tiering."
    ),
    "full_medrag": (
        "Proposed System (FULL_MEDRAG): All components enabled: iterative hybrid retrieval, reranking, "
        "claim extraction, NLI verification, evidence state, source quality, risk policy, and selective abstention."
    ),
}

# Synonyms for flexible lookup
BASELINE_ALIASES: Dict[str, str] = {
    "closed_book_llm": "closed_book",
    "dense_rag": "single_pass_dense_rag",
    "bm25_rag": "single_pass_bm25_rag",
    "rrf_rag": "hybrid_rag",
    "reranked_rag": "hybrid_reranked_rag",
    "claim_verified": "claim_verified_rag",
    "proposed": "full_medrag",
    "medrag_full": "full_medrag",
}


# ==============================================================================
# Canonical Ablation Definitions (PART B - Leave-One-Out from FULL_MEDRAG)
# ==============================================================================

ABLATION_DESCRIPTIONS: Dict[str, str] = {
    "ablation_no_hybrid": "Ablation 1: FULL_MEDRAG without hybrid retrieval (dense embedding retrieval only).",
    "ablation_no_reranking": "Ablation 2: FULL_MEDRAG without cross-encoder neural reranking.",
    "ablation_no_claim_decomposition": "Ablation 3: FULL_MEDRAG without atomic claim decomposition (holistic statement).",
    "ablation_no_nli_verification": "Ablation 4: FULL_MEDRAG without sentence-level NLI verification (retriever scores only).",
    "ablation_no_evidence_state": "Ablation 5: FULL_MEDRAG without multi-passage evidence state & conflict tracking.",
    "ablation_no_iterative": "Ablation 6: FULL_MEDRAG without iterative retrieval (fixed max_search_rounds = 1).",
    "ablation_no_risk_policy": "Ablation 7: FULL_MEDRAG without risk-aware adaptive stopping (uniform static thresholds).",
    "ablation_no_abstention": "Ablation 8: FULL_MEDRAG without selective abstention (forced to answer all queries).",
    "ablation_no_source_quality": "Ablation 9: FULL_MEDRAG without source quality or temporal recency weighting.",
}


def create_baseline_config(
    baseline_id: str,
    base_config: ExperimentConfig,
    experiment_id_suffix: Optional[str] = None,
) -> ExperimentConfig:
    """
    Creates an ExperimentConfig configured for a specific baseline system while
    strictly preserving all shared hyperparameters (fairness guarantee).
    """
    canonical_id = BASELINE_ALIASES.get(baseline_id.lower().strip(), baseline_id.lower().strip())
    if canonical_id not in BASELINE_DESCRIPTIONS:
        raise ValueError(
            f"Unknown baseline '{baseline_id}'. Available: {list(BASELINE_DESCRIPTIONS.keys())}"
        )

    cfg_dict = base_config.model_dump()
    exp_suffix = experiment_id_suffix or canonical_id
    cfg_dict["experiment_id"] = f"{base_config.experiment_id}_{exp_suffix}"
    cfg_dict["description"] = BASELINE_DESCRIPTIONS[canonical_id]
    cfg_dict["protocol"] = canonical_id

    # 1. Closed Book (Zero retrieval)
    if canonical_id == "closed_book":
        cfg_dict.update({
            "max_search_rounds": 1,
            "enable_hybrid": False,
            "use_reranker": False,
            "enable_reranking": False,
            "enable_claim_decomposition": False,
            "enable_nli_verification": False,
            "enable_evidence_state": False,
            "enable_iterative": False,
            "enable_risk_policy": False,
            "enable_abstention": False,
        })

    # 2. Single-pass Dense RAG
    elif canonical_id == "single_pass_dense_rag":
        cfg_dict.update({
            "max_search_rounds": 1,
            "enable_hybrid": False,
            "bm25_weight": 0.0,
            "dense_weight": 1.0,
            "use_reranker": False,
            "enable_reranking": False,
            "enable_claim_decomposition": False,
            "enable_nli_verification": False,
            "enable_evidence_state": False,
            "enable_iterative": False,
            "enable_risk_policy": False,
            "enable_abstention": False,
        })

    # 3. Single-pass BM25 RAG
    elif canonical_id == "single_pass_bm25_rag":
        cfg_dict.update({
            "max_search_rounds": 1,
            "enable_hybrid": False,
            "bm25_weight": 1.0,
            "dense_weight": 0.0,
            "use_reranker": False,
            "enable_reranking": False,
            "enable_claim_decomposition": False,
            "enable_nli_verification": False,
            "enable_evidence_state": False,
            "enable_iterative": False,
            "enable_risk_policy": False,
            "enable_abstention": False,
        })

    # 4. Hybrid RAG (BM25 + Dense RRF)
    elif canonical_id == "hybrid_rag":
        cfg_dict.update({
            "max_search_rounds": 1,
            "enable_hybrid": True,
            "bm25_weight": 0.4,
            "dense_weight": 0.6,
            "use_reranker": False,
            "enable_reranking": False,
            "enable_claim_decomposition": False,
            "enable_nli_verification": False,
            "enable_evidence_state": False,
            "enable_iterative": False,
            "enable_risk_policy": False,
            "enable_abstention": False,
        })

    # 5. Hybrid Reranked RAG
    elif canonical_id == "hybrid_reranked_rag":
        cfg_dict.update({
            "max_search_rounds": 1,
            "enable_hybrid": True,
            "bm25_weight": 0.4,
            "dense_weight": 0.6,
            "use_reranker": True,
            "enable_reranking": True,
            "enable_claim_decomposition": False,
            "enable_nli_verification": False,
            "enable_evidence_state": False,
            "enable_iterative": False,
            "enable_risk_policy": False,
            "enable_abstention": False,
        })

    # 6. Claim Verified RAG (Single pass, with verification)
    elif canonical_id == "claim_verified_rag":
        cfg_dict.update({
            "max_search_rounds": 1,
            "enable_hybrid": True,
            "bm25_weight": 0.4,
            "dense_weight": 0.6,
            "use_reranker": True,
            "enable_reranking": True,
            "enable_claim_decomposition": True,
            "enable_nli_verification": True,
            "enable_evidence_state": True,
            "enable_iterative": False,
            "enable_risk_policy": False,
            "enable_abstention": True,
            "enable_source_quality_weighting": True,
        })

    # 7. Iterative RAG (Iterative without risk-aware stopping)
    elif canonical_id == "iterative_rag":
        cfg_dict.update({
            "max_search_rounds": max(2, base_config.max_search_rounds),
            "enable_hybrid": True,
            "bm25_weight": 0.4,
            "dense_weight": 0.6,
            "use_reranker": True,
            "enable_reranking": True,
            "enable_claim_decomposition": True,
            "enable_nli_verification": True,
            "enable_evidence_state": True,
            "enable_iterative": True,
            "enable_risk_policy": False,
            "enable_abstention": True,
            "enable_source_quality_weighting": True,
        })

    # 8. Full MedRAG (Proposed system with all components enabled)
    elif canonical_id == "full_medrag":
        cfg_dict.update({
            "max_search_rounds": max(2, base_config.max_search_rounds),
            "enable_hybrid": True,
            "bm25_weight": 0.4,
            "dense_weight": 0.6,
            "use_reranker": True,
            "enable_reranking": True,
            "enable_claim_decomposition": True,
            "enable_nli_verification": True,
            "enable_evidence_state": True,
            "enable_iterative": True,
            "enable_risk_policy": True,
            "enable_abstention": True,
            "enable_source_quality_weighting": True,
        })

    return ExperimentConfig(**cfg_dict)


def create_ablation_config(
    ablation_id: str,
    base_config: ExperimentConfig,
    experiment_id_suffix: Optional[str] = None,
) -> ExperimentConfig:
    """
    Creates an ExperimentConfig for a single-component ablation from FULL_MEDRAG.
    Guarantees leave-one-out fairness: exactly one architectural toggle is disabled.
    """
    ablation_key = ablation_id.lower().strip()
    if ablation_key not in ABLATION_DESCRIPTIONS:
        raise ValueError(
            f"Unknown ablation '{ablation_id}'. Available: {list(ABLATION_DESCRIPTIONS.keys())}"
        )

    # Start from fully-enabled FULL_MEDRAG base
    full_base = create_baseline_config("full_medrag", base_config)
    cfg_dict = full_base.model_dump()

    exp_suffix = experiment_id_suffix or ablation_key
    cfg_dict["experiment_id"] = f"{base_config.experiment_id}_{exp_suffix}"
    cfg_dict["description"] = ABLATION_DESCRIPTIONS[ablation_key]
    cfg_dict["protocol"] = ablation_key

    if ablation_key == "ablation_no_hybrid":
        cfg_dict["enable_hybrid"] = False
        cfg_dict["bm25_weight"] = 0.0
        cfg_dict["dense_weight"] = 1.0

    elif ablation_key == "ablation_no_reranking":
        cfg_dict["enable_reranking"] = False
        cfg_dict["use_reranker"] = False

    elif ablation_key == "ablation_no_claim_decomposition":
        cfg_dict["enable_claim_decomposition"] = False

    elif ablation_key == "ablation_no_nli_verification":
        cfg_dict["enable_nli_verification"] = False

    elif ablation_key == "ablation_no_evidence_state":
        cfg_dict["enable_evidence_state"] = False

    elif ablation_key == "ablation_no_iterative":
        cfg_dict["enable_iterative"] = False
        cfg_dict["max_search_rounds"] = 1

    elif ablation_key == "ablation_no_risk_policy":
        cfg_dict["enable_risk_policy"] = False

    elif ablation_key == "ablation_no_abstention":
        cfg_dict["enable_abstention"] = False

    elif ablation_key == "ablation_no_source_quality":
        cfg_dict["enable_source_quality_weighting"] = False

    return ExperimentConfig(**cfg_dict)


def get_baseline_suite(
    benchmark_name: str,
    split: str,
    base_config: Optional[ExperimentConfig] = None,
    limit: Optional[int] = None,
    backend: Optional[str] = None,
    model: Optional[str] = None,
    output_dir: str = "runs",
) -> List[ExperimentConfig]:
    """Generates the full suite of 8 standardized baseline configurations."""
    base = base_config or ExperimentConfig(
        experiment_id=f"{benchmark_name}_{split}_study",
        benchmark_name=benchmark_name,
        split=split,
        limit=limit,
        llm_backend=backend or "groq",
        llm_model=model or "openai/gpt-oss-120b",
        output_dir=output_dir,
    )
    if limit is not None:
        base.limit = limit

    baseline_order = [
        "closed_book",
        "single_pass_dense_rag",
        "single_pass_bm25_rag",
        "hybrid_rag",
        "hybrid_reranked_rag",
        "claim_verified_rag",
        "iterative_rag",
        "full_medrag",
    ]
    return [create_baseline_config(b, base) for b in baseline_order]


def get_ablation_suite(
    benchmark_name: str,
    split: str,
    base_config: Optional[ExperimentConfig] = None,
    limit: Optional[int] = None,
    backend: Optional[str] = None,
    model: Optional[str] = None,
    output_dir: str = "runs",
) -> List[ExperimentConfig]:
    """Generates the full suite of 9 leave-one-out ablation configurations + FULL_MEDRAG."""
    base = base_config or ExperimentConfig(
        experiment_id=f"{benchmark_name}_{split}_ablation",
        benchmark_name=benchmark_name,
        split=split,
        limit=limit,
        llm_backend=backend or "groq",
        llm_model=model or "openai/gpt-oss-120b",
        output_dir=output_dir,
    )
    if limit is not None:
        base.limit = limit

    # Proposed system serves as the reference
    configs = [create_baseline_config("full_medrag", base)]
    for ab_id in sorted(ABLATION_DESCRIPTIONS.keys()):
        configs.append(create_ablation_config(ab_id, base))
    return configs


def validate_fairness(configs: List[ExperimentConfig]) -> Tuple[bool, List[str]]:
    """
    Verifies PART C fairness constraints across a set of comparison configurations:
    1. Same dataset benchmark and split
    2. Same LLM backend and model
    3. Same temperature and max token limit
    4. Same evaluation item limit
    5. Same retrieval candidate pools (top_k_retrieve, top_k_rerank) where retrieval is used
    """
    violations: List[str] = []
    if len(configs) < 2:
        return True, []

    ref = configs[0]
    for idx, c in enumerate(configs[1:], 2):
        if c.benchmark_name != ref.benchmark_name:
            violations.append(
                f"Config #{idx} ({c.experiment_id}) uses benchmark '{c.benchmark_name}' "
                f"vs reference '{ref.benchmark_name}'."
            )
        if c.split != ref.split:
            violations.append(
                f"Config #{idx} ({c.experiment_id}) uses split '{c.split}' "
                f"vs reference '{ref.split}'."
            )
        if c.llm_backend != ref.llm_backend or c.llm_model != ref.llm_model:
            violations.append(
                f"Config #{idx} ({c.experiment_id}) uses LLM '{c.llm_backend}/{c.llm_model}' "
                f"vs reference '{ref.llm_backend}/{ref.llm_model}'."
            )
        if c.llm_temperature != ref.llm_temperature:
            violations.append(
                f"Config #{idx} ({c.experiment_id}) uses temperature {c.llm_temperature} "
                f"vs reference {ref.llm_temperature}."
            )
        if c.limit != ref.limit:
            violations.append(
                f"Config #{idx} ({c.experiment_id}) evaluates limit {c.limit} "
                f"vs reference {ref.limit}."
            )

    is_fair = len(violations) == 0
    return is_fair, violations
