"""
MedRAG Benchmark Evaluation Framework.
Provides reusable medical dataset adapters, evaluation execution,
reproducible manifests, streaming artifact storage, and scientific metrics.
"""
from .schema import BenchmarkExample, PredictionRecord, RetrievalTrace
from .adapter import DatasetAdapter
from .registry import BenchmarkRegistry
from .manifest import ExperimentManifest
from .recorder import PredictionRecorder
from .metrics import MetricRegistry, EvaluationMetrics
from .runner import EvaluationRunner
from ..config_schema import ExperimentConfig

from .experiments import (
    BASELINE_DESCRIPTIONS,
    ABLATION_DESCRIPTIONS,
    create_baseline_config,
    create_ablation_config,
    get_baseline_suite,
    get_ablation_suite,
    validate_fairness,
)
from .significance import (
    McNemarResult,
    BootstrapResult,
    PairedComparisonReport,
    compute_mcnemar_test,
    paired_bootstrap_ci,
    compare_systems,
)
from .study import StudyRunner, StudyResult

from .claims import (
    ClaimEvidenceLabel,
    ClaimAnnotation,
    ClaimEvaluationMetrics,
    evaluate_claim_faithfulness,
    load_claim_annotations,
    save_claim_annotations,
)
from .abstention import (
    RiskCoveragePoint,
    AbstentionClassificationMetrics,
    AbstentionEvaluationReport,
    ThresholdTuningResult,
    compute_risk_coverage_curve,
    evaluate_abstention_classification,
    tune_abstention_thresholds,
    evaluate_abstention,
)
from .calibration import (
    CalibrationBin,
    CalibrationMetrics,
    TemperatureScaler,
    compute_brier_score,
    compute_expected_calibration_error,
    evaluate_calibration,
)
from .statistical_report import (
    MetricStatistic,
    ComprehensiveStatisticalReport,
    compute_wilson_score_interval,
    compute_bootstrap_ci,
    generate_comprehensive_statistical_report,
)
from .visualizations import (
    generate_all_publication_figures,
    plot_baseline_accuracy_comparison,
    plot_retrieval_recall_at_k,
    plot_claim_verification_performance,
    plot_confusion_matrix,
    plot_ablation_results,
    plot_risk_coverage_curve,
    plot_reliability_diagram,
    plot_latency_vs_accuracy,
    plot_retrieval_round_distribution,
)

# Automatically register built-in adapters
from .adapters import SyntheticAdapter, MedQAAdapter, PubMedQAAdapter

__all__ = [
    "BenchmarkExample",
    "PredictionRecord",
    "RetrievalTrace",
    "DatasetAdapter",
    "BenchmarkRegistry",
    "ExperimentManifest",
    "PredictionRecorder",
    "MetricRegistry",
    "EvaluationMetrics",
    "EvaluationRunner",
    "ExperimentConfig",
    "SyntheticAdapter",
    "MedQAAdapter",
    "PubMedQAAdapter",
    "BASELINE_DESCRIPTIONS",
    "ABLATION_DESCRIPTIONS",
    "create_baseline_config",
    "create_ablation_config",
    "get_baseline_suite",
    "get_ablation_suite",
    "validate_fairness",
    "McNemarResult",
    "BootstrapResult",
    "PairedComparisonReport",
    "compute_mcnemar_test",
    "paired_bootstrap_ci",
    "compare_systems",
    "StudyRunner",
    "StudyResult",
    "ClaimEvidenceLabel",
    "ClaimAnnotation",
    "ClaimEvaluationMetrics",
    "evaluate_claim_faithfulness",
    "load_claim_annotations",
    "save_claim_annotations",
    "RiskCoveragePoint",
    "AbstentionClassificationMetrics",
    "AbstentionEvaluationReport",
    "ThresholdTuningResult",
    "compute_risk_coverage_curve",
    "evaluate_abstention_classification",
    "tune_abstention_thresholds",
    "evaluate_abstention",
    "CalibrationBin",
    "CalibrationMetrics",
    "TemperatureScaler",
    "compute_brier_score",
    "compute_expected_calibration_error",
    "evaluate_calibration",
    "MetricStatistic",
    "ComprehensiveStatisticalReport",
    "compute_wilson_score_interval",
    "compute_bootstrap_ci",
    "generate_comprehensive_statistical_report",
    "generate_all_publication_figures",
    "plot_baseline_accuracy_comparison",
    "plot_retrieval_recall_at_k",
    "plot_claim_verification_performance",
    "plot_confusion_matrix",
    "plot_ablation_results",
    "plot_risk_coverage_curve",
    "plot_reliability_diagram",
    "plot_latency_vs_accuracy",
    "plot_retrieval_round_distribution",
]
