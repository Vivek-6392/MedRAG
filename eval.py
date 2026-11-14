#!/usr/bin/env python
"""
MedRAG Benchmark Evaluation CLI.
Launches, resumes, and evaluates reproducible medical QA benchmark experiments.
Usable standalone without Streamlit.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Environment safety & offline resilience
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
if hf_cache.exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from medrag.benchmarks import (
    BenchmarkRegistry,
    EvaluationMetrics,
    EvaluationRunner,
    ExperimentConfig,
    StudyRunner,
    get_baseline_suite,
    get_ablation_suite,
    load_claim_annotations,
    evaluate_claim_faithfulness,
    tune_abstention_thresholds,
    generate_all_publication_figures,
)
from medrag.logger import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MedRAG Benchmark Evaluation CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core execution flags
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML or JSON experiment configuration file (e.g. configs/medqa_eval.yaml)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Quick smoke test: runs 3 synthetic examples with mock backend to verify pipeline integrity",
    )
    parser.add_argument(
        "--list-benchmarks",
        action="store_true",
        help="List all registered benchmark datasets and exit",
    )

    # Protocol & Study selection
    parser.add_argument(
        "--protocol",
        type=str,
        default="full_medrag",
        help=(
            "Evaluation protocol or baseline: 'closed_book', 'single_pass_dense_rag', "
            "'single_pass_bm25_rag', 'hybrid_rag', 'hybrid_reranked_rag', 'claim_verified_rag', "
            "'iterative_rag', 'full_medrag', 'pubmedqa_provided_context', 'pubmedqa_retrieval', "
            "or an ablation preset (e.g. 'ablation_no_hybrid', 'ablation_no_reranking')"
        ),
    )
    parser.add_argument(
        "--study",
        type=str,
        default=None,
        choices=["baselines", "ablations"],
        help="Run an automated multi-configuration study suite ('baselines' or 'ablations')",
    )

    # Dataset & Split selection
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        help="Target benchmark dataset ('synthetic', 'medqa', 'pubmedqa')",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split ('test', 'val', 'dev', 'train')",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate (for smoke testing / quick validation)",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Path to custom knowledge corpus JSON/JSONL file (optional)",
    )
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default=None,
        help="Directory containing medical textbook .txt files to chunk and index (optional)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Textbook chunk size in characters",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="Textbook chunk overlap in characters",
    )

    # Experiment metadata & resume
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Unique experiment identifier (e.g. 'medqa_eval_v1'). Defaults to auto-generated ID.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs",
        help="Root directory for saving experiment runs and artifacts",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume; forces re-evaluating completed examples from scratch",
    )

    # Model configuration
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["mock", "groq", "openai", "ollama"],
        help="LLM backend. Defaults to 'mock' if credentials are unset",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model identifier (e.g. 'llama-3.3-70b-versatile', 'gpt-4o-mini')",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM sampling temperature",
    )

    # Retriever & Verifier configuration
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="Maximum iterative retrieval rounds",
    )
    parser.add_argument(
        "--support-threshold",
        type=float,
        default=0.70,
        help="Claim verification support threshold",
    )
    parser.add_argument(
        "--conflict-threshold",
        type=float,
        default=0.35,
        help="Contradiction detection threshold",
    )
    parser.add_argument(
        "--high-risk-threshold",
        type=float,
        default=0.75,
        help="Clinical high-risk threshold for conservative abstention",
    )

    # Research Analysis & Visualizations
    parser.add_argument(
        "--generate-plots",
        "--plots",
        action="store_true",
        help="Generate all 9 publication-grade figures in publication/figures/",
    )
    parser.add_argument(
        "--claims-eval",
        type=str,
        default=None,
        help="Path to gold claim annotations JSON/JSONL file to evaluate atomic claim faithfulness",
    )
    parser.add_argument(
        "--tune-thresholds",
        action="store_true",
        help="Tune abstention confidence/evidence thresholds on validation split",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = get_logger("medrag.eval.cli")

    if args.list_benchmarks:
        benchmarks = BenchmarkRegistry.list_available()
        print("\nRegistered MedRAG Benchmark Adapters:")
        for b in benchmarks:
            adapter = BenchmarkRegistry.get(b)
            print(f"  • {b:<12} (version: {adapter.version}, splits: {adapter.supported_splits})")
        print()
        return 0

    # Smoke test mode shortcut
    if args.smoke_test:
        dataset = "synthetic"
        split = "test"
        limit = 3
        protocol = "medrag_full"
        backend = "mock"
        model = "mock-clinical-llm"
        exp_id = f"smoke_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Running automated smoke test: {exp_id}")
    elif args.config:
        config = ExperimentConfig.from_file(args.config)
        dataset = config.benchmark_name
        split = args.split if args.split != "test" else config.split
        limit = args.limit if args.limit is not None else config.limit
        protocol = args.protocol if args.protocol != "full_medrag" else config.protocol
        backend = args.backend or config.llm_backend
        model = args.model or config.llm_model
        exp_id = args.experiment_id or config.experiment_id
        logger.info(f"Loaded experiment configuration from '{args.config}': {config.experiment_id}")
    else:
        dataset = args.dataset
        split = args.split
        limit = args.limit
        protocol = args.protocol
        backend = args.backend or ("mock" if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("GROQ_API_KEY") else "groq")
        model = args.model or ("mock-clinical-llm" if backend == "mock" else "llama-3.3-70b-versatile")
        exp_id = args.experiment_id or f"{dataset}_{protocol}_{split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Handle textbook corpus preparation if requested
    prepared_docs = None
    if args.corpus_dir:
        from medrag.benchmarks.corpus_preparer import TextbookCorpusPreparer
        preparer = TextbookCorpusPreparer(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        logger.info(f"Preparing textbook corpus from {args.corpus_dir}...")
        prepared_docs, manifest = preparer.prepare_directory(args.corpus_dir)
        logger.info(f"Loaded {len(prepared_docs)} chunks (Corpus SHA256: {manifest.get('corpus_sha256', '')[:12]}...)")

    if not args.config:
        config = ExperimentConfig(
            experiment_id=exp_id,
            benchmark_name=dataset,
            split=split,
            protocol=protocol,
            limit=limit,
            resume=not args.no_resume,
            corpus_path=args.corpus,
            output_dir=args.output_dir,
            llm_backend=backend,
            llm_model=model,
            llm_temperature=args.temperature,
            max_search_rounds=args.max_rounds,
            support_threshold=args.support_threshold,
            conflict_threshold=args.conflict_threshold,
            high_risk_threshold=args.high_risk_threshold,
        )
    else:
        # Apply any explicit CLI overrides to the loaded configuration
        if args.limit is not None:
            config.limit = args.limit
        if args.split != "test":
            config.split = args.split
        if args.experiment_id:
            config.experiment_id = args.experiment_id
        if args.output_dir != "runs":
            config.output_dir = args.output_dir
        if args.corpus:
            config.corpus_path = args.corpus

    # Multi-configuration study execution mode
    if args.study:
        study_id = exp_id
        if args.study == "baselines":
            study_configs = get_baseline_suite(
                benchmark_name=dataset,
                split=split,
                base_config=config,
                limit=limit,
                backend=backend,
                model=model,
                output_dir=args.output_dir,
            )
        elif args.study == "ablations":
            study_configs = get_ablation_suite(
                benchmark_name=dataset,
                split=split,
                base_config=config,
                limit=limit,
                backend=backend,
                model=model,
                output_dir=args.output_dir,
            )
        else:
            raise ValueError(f"Unknown study suite '{args.study}'")

        print("\n" + "=" * 70)
        print(f"  MedRAG Automated Study Suite: {args.study.upper()}")
        print(f"  Study ID      : {study_id}")
        print(f"  Dataset       : {dataset} ({split})")
        print(f"  Configurations: {len(study_configs)} total")
        print(f"  Limit         : {limit or 'All'}")
        print(f"  Backend       : {backend} ({model})")
        print(f"  Output Dir    : {os.path.join(args.output_dir, study_id)}")
        print("=" * 70 + "\n")

        study_runner = StudyRunner(
            study_id=study_id,
            configs=study_configs,
            corpus=prepared_docs,
            output_dir=args.output_dir,
        )
        study_result = study_runner.run()

        print("\n" + "=" * 70)
        print(study_result.comparison_table_md)
        if study_result.ablation_table_md:
            print("\n" + study_result.ablation_table_md)
        print("=" * 70 + "\n")
        print(f"Study results and per-example predictions saved to: {study_result.output_dir}\n")
        return 0

    print("\n" + "=" * 70)
    print("  MedRAG Benchmark Evaluation Engine")
    print(f"  Experiment ID : {config.experiment_id}")
    print(f"  Dataset       : {config.benchmark_name} ({config.split})")
    print(f"  Protocol      : {config.protocol.upper()}")
    print(f"  Limit         : {config.limit or 'All'}")
    print(f"  Backend       : {config.llm_backend} ({config.llm_model})")
    print(f"  Output Dir    : {os.path.join(config.output_dir, config.experiment_id)}")
    print("=" * 70 + "\n")

    try:
        runner = EvaluationRunner(config=config, corpus=prepared_docs, logger=logger)
        metrics: EvaluationMetrics = runner.run()
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        print(f"\n[ERROR] Experiment failed: {e}\n")
        return 1

    print("\n" + "=" * 70)
    print(f"  EVALUATION RESULTS — {config.experiment_id}")
    print("=" * 70)
    print(f"  Protocol            : {config.protocol.upper()}")
    print(f"  Total Examples      : {metrics.total_examples}")
    print(f"  Evaluated (Answered): {metrics.evaluated_count}")
    print(f"  Abstained           : {metrics.abstained_count}")
    print(f"  Raw Accuracy        : {metrics.accuracy * 100:.2f}%")
    print(f"  Selective Accuracy  : {metrics.selective_accuracy * 100:.2f}%")
    if metrics.macro_f1 is not None:
        print(f"  Macro-F1 Score      : {metrics.macro_f1:.4f}")
    print(f"  Abstention Rate     : {metrics.abstention_rate * 100:.2f}%")
    print(f"  Coverage            : {metrics.coverage * 100:.2f}%")
    if metrics.per_option_accuracy:
        po_str = ", ".join(f"{k}: {v * 100:.1f}%" for k, v in sorted(metrics.per_option_accuracy.items()))
        print(f"  Per-Option Accuracy : {po_str}")
    if metrics.aurc is not None:
        print(f"  AURC Risk-Coverage  : {metrics.aurc:.4f}")
    print(f"  Mean Latency        : {metrics.mean_latency_seconds:.2f}s")
    print(f"  Mean Search Rounds  : {metrics.mean_retrieval_rounds:.2f}")
    if metrics.mean_total_tokens > 0:
        print(f"  Mean Tokens (Total) : {metrics.mean_total_tokens:.1f} (prompt={metrics.mean_prompt_tokens:.1f}, completion={metrics.mean_completion_tokens:.1f})")
    if metrics.confusion_matrix:
        print("\n  Confusion Matrix (Gold Rows x Predicted Cols):")
        cols = sorted(list(next(iter(metrics.confusion_matrix.values())).keys()))
        header = "    Gold \\ Pred  " + "  ".join(f"{c:>7}" for c in cols)
        print(header)
        for g_row in sorted(metrics.confusion_matrix.keys()):
            row_vals = "  ".join(f"{metrics.confusion_matrix[g_row].get(c, 0):>7}" for c in cols)
            print(f"    {g_row:<12} {row_vals}")
    print("=" * 70)
    print(f"  Run artifacts saved to: {os.path.abspath(os.path.join(config.output_dir, config.experiment_id))}\n")

    # Post-evaluation claim-level analysis
    all_recs = runner.recorder.load_all_predictions()
    if args.claims_eval:
        print("\n" + "=" * 70)
        print("  CLAIM-LEVEL FAITHFULNESS EVALUATION (Annotated Subset)")
        print("=" * 70)
        try:
            anns = load_claim_annotations(args.claims_eval)
            claim_metrics = evaluate_claim_faithfulness(all_recs, anns, subset_only=True)
            print(f"  Annotated Subset Size : {claim_metrics.annotated_subset_evaluated} claims")
            print(f"  Macro Precision       : {claim_metrics.precision * 100:.2f}%")
            print(f"  Macro Recall          : {claim_metrics.recall * 100:.2f}%")
            print(f"  Macro-F1 Score        : {claim_metrics.macro_f1:.4f}")
            print(f"  Unsupported Claim Rate: {claim_metrics.unsupported_claim_rate * 100:.2f}%")
            print(f"  False-Support Rate    : {claim_metrics.false_support_rate * 100:.2f}%")
            print(f"  Citation Precision    : {claim_metrics.citation_precision * 100:.2f}%")
            print(f"  Citation Completeness : {claim_metrics.citation_completeness * 100:.2f}%")
            print(f"\n  Notice: {claim_metrics.annotation_subset_warning}")
            print("=" * 70 + "\n")
        except Exception as e:
            logger.error(f"Failed to evaluate claim faithfulness: {e}")

    # Validation threshold tuning
    if args.tune_thresholds:
        print("\n" + "=" * 70)
        print("  VALIDATION ABSTENTION THRESHOLD TUNING")
        print("=" * 70)
        tune_res = tune_abstention_thresholds(all_recs, target_risk=0.05, min_coverage=0.50)
        print(f"  Optimal Threshold     : {tune_res.optimal_threshold:.4f}")
        print(f"  Validation Coverage   : {tune_res.val_coverage * 100:.1f}%")
        print(f"  Validation Risk       : {tune_res.val_selective_risk * 100:.2f}%")
        print(f"  Validation Accuracy   : {tune_res.val_selective_accuracy * 100:.2f}%")
        print(f"  Methodology Note      : {tune_res.methodology_note}")
        print("=" * 70 + "\n")

    # Generate publication plots if requested
    if args.generate_plots:
        print("\n" + "=" * 70)
        print("  GENERATING PUBLICATION-QUALITY FIGURES")
        print("=" * 70)
        fig_map = generate_all_publication_figures(
            metrics_by_system={config.experiment_id: metrics},
            records_by_system={config.experiment_id: all_recs},
            output_dir=os.path.join("publication", "figures"),
        )
        for fig_name, fig_path in sorted(fig_map.items()):
            print(f"  • {fig_name:<22}: {fig_path}")
        print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
