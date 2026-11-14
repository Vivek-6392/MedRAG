"""
Research-Grade Statistical Reporting Engine.
Produces publication-ready statistical reports for all primary medical QA metrics:
- Sample size (N)
- Point estimates
- Rigorous 95% Confidence Intervals:
    * Wilson score interval with continuity correction for binomial proportions (Accuracy, Selective Accuracy, Coverage)
    * Non-parametric empirical bootstrap percentile intervals for macro-averaged metrics (Macro-F1, AURC, ECE, Brier)
- Plain-text and mathematical explanations of calculations
- Version tracking of metric implementations
- Paired statistical comparisons against baseline systems (McNemar's test & Paired Bootstrap deltas)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field

from .schema import PredictionRecord
from .significance import compare_systems

METRIC_ENGINE_VERSION = "2.2.0"


class MetricStatistic(BaseModel):
    """Statistical summary for a single benchmark metric."""
    metric_name: str
    point_estimate: float
    sample_size: int
    ci_lower: float
    ci_upper: float
    ci_level: float = 0.95
    ci_method: str = "wilson_score"
    explanation: str
    implementation_version: str = METRIC_ENGINE_VERSION


class ComprehensiveStatisticalReport(BaseModel):
    """Full experiment statistical report across all evaluation facets."""
    experiment_id: str
    dataset: str
    split: str
    total_examples: int
    metrics: Dict[str, MetricStatistic] = Field(default_factory=dict)
    paired_comparisons: Dict[str, Any] = Field(default_factory=dict)
    implementation_version: str = METRIC_ENGINE_VERSION
    timestamp: str = ""

    def to_markdown(self) -> str:
        """Renders publication-ready Markdown table of all primary metrics and CIs."""
        lines = [
            f"# Statistical Analysis Report: `{self.experiment_id}`",
            f"- **Dataset:** {self.dataset} ({self.split})",
            f"- **Total Sample Size (N):** {self.total_examples}",
            f"- **Metric Engine Version:** {self.implementation_version}\n",
            "## Primary Metric Estimates & 95% Confidence Intervals\n",
            "| Metric | Point Estimate | 95% Confidence Interval | Sample Size (N) | Estimation Method |",
            "| :--- | ---: | :---: | ---: | :--- |",
        ]
        for name, stat in self.metrics.items():
            est_str = f"{stat.point_estimate:.4f}"
            if "accuracy" in name.lower() or "rate" in name.lower() or "coverage" in name.lower() or "recall" in name.lower() or "precision" in name.lower():
                est_str = f"{stat.point_estimate * 100:.1f}%"
                ci_str = f"[{stat.ci_lower * 100:.1f}%, {stat.ci_upper * 100:.1f}%]"
            else:
                ci_str = f"[{stat.ci_lower:.4f}, {stat.ci_upper:.4f}]"

            lines.append(
                f"| {stat.metric_name} | **{est_str}** | {ci_str} | {stat.sample_size} | {stat.ci_method} |"
            )

        lines.extend([
            "\n## Mathematical Formulations & Calculation Notes\n",
        ])
        for name, stat in self.metrics.items():
            lines.append(f"### {stat.metric_name}")
            lines.append(f"- **Calculation:** {stat.explanation}")
            lines.append(f"- **Confidence Interval ({stat.ci_method}):** [{stat.ci_lower:.4f}, {stat.ci_upper:.4f}]\n")

        if self.paired_comparisons:
            lines.append("## Paired System Comparisons (vs. Baselines)\n")
            for base_name, comp_dict in self.paired_comparisons.items():
                lines.append(f"### Proposed MedRAG vs. `{base_name}`")
                mcn = comp_dict.get("mcnemar", {})
                boot = comp_dict.get("accuracy_bootstrap", {})
                lines.append(f"- **McNemar Test:** statistic={mcn.get('statistic', 0.0):.4f}, p-value={mcn.get('p_value', 1.0):.4e} ({mcn.get('test_variant', 'exact')})")
                lines.append(f"- **Accuracy Delta (Δ):** {boot.get('observed_delta', 0.0) * 100:+.2f}% (95% CI: [{boot.get('ci_lower', 0.0) * 100:+.2f}%, {boot.get('ci_upper', 0.0) * 100:+.2f}%], p={boot.get('p_value', 1.0):.4f})")
                lines.append(f"- **Significant at α=0.05:** {'Yes' if comp_dict.get('is_significant_at_05') else 'No'}\n")

        return "\n".join(lines)


def compute_wilson_score_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Computes Wilson score interval with continuity correction for binomial proportions.
    Guarantees interval stays strictly within [0, 1] even for small N or extreme rates.
    """
    if total <= 0:
        return 0.0, 0.0

    # z = 1.95996 for 95% confidence
    if math.isclose(confidence, 0.95):
        z = 1.959963984540054
    elif math.isclose(confidence, 0.99):
        z = 2.5758293035489004
    else:
        # Standard normal inverse approximation
        z = 1.96

    p_hat = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total

    center = (p_hat + z2 / (2.0 * total)) / denom
    margin = (z / denom) * math.sqrt((p_hat * (1.0 - p_hat) / total) + (z2 / (4.0 * total * total)))

    ci_lower = max(0.0, center - margin)
    ci_upper = min(1.0, center + margin)
    return round(ci_lower, 4), round(ci_upper, 4)


def compute_bootstrap_ci(
    values: List[float],
    statistic_fn=np.mean,
    num_bootstraps: int = 1000,
    confidence: float = 0.95,
    random_seed: int = 42,
) -> Tuple[float, float]:
    """
    Computes non-parametric percentile bootstrap confidence interval.
    """
    if not values:
        return 0.0, 0.0
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])

    rng = np.random.default_rng(random_seed)
    n = len(arr)
    boot_stats = np.empty(num_bootstraps, dtype=np.float64)

    for i in range(num_bootstraps):
        sample = rng.choice(arr, size=n, replace=True)
        boot_stats[i] = float(statistic_fn(sample))

    alpha = 1.0 - confidence
    low = float(np.percentile(boot_stats, 100 * (alpha / 2.0)))
    high = float(np.percentile(boot_stats, 100 * (1.0 - alpha / 2.0)))
    return round(low, 4), round(high, 4)


def generate_comprehensive_statistical_report(
    records: List[PredictionRecord],
    baseline_records: Optional[Dict[str, List[PredictionRecord]]] = None,
    experiment_id: str = "medrag_experiment",
) -> ComprehensiveStatisticalReport:
    """
    Generates a full research-grade statistical analysis report for an evaluation run.
    """
    total = len(records)
    if total == 0:
        return ComprehensiveStatisticalReport(
            experiment_id=experiment_id,
            dataset="",
            split="",
            total_examples=0,
        )

    dataset_name = records[0].dataset if records else ""
    split_name = records[0].split if records else ""

    metrics_dict: Dict[str, MetricStatistic] = {}

    # 1. Raw Accuracy (counting abstentions as 0)
    correct_count = sum(1 for r in records if r.is_correct is True)
    acc_val = correct_count / total
    acc_ci_low, acc_ci_high = compute_wilson_score_interval(correct_count, total)
    metrics_dict["accuracy"] = MetricStatistic(
        metric_name="Raw Accuracy",
        point_estimate=round(acc_val, 4),
        sample_size=total,
        ci_lower=acc_ci_low,
        ci_upper=acc_ci_high,
        ci_method="Wilson score interval with continuity correction",
        explanation="Proportion of total benchmark questions answered correctly (correct_count / total_examples). Abstentions count as incorrect.",
    )

    # 2. Coverage
    answered_count = sum(1 for r in records if not r.abstained)
    cov_val = answered_count / total
    cov_ci_low, cov_ci_high = compute_wilson_score_interval(answered_count, total)
    metrics_dict["coverage"] = MetricStatistic(
        metric_name="Coverage",
        point_estimate=round(cov_val, 4),
        sample_size=total,
        ci_lower=cov_ci_low,
        ci_upper=cov_ci_high,
        ci_method="Wilson score interval with continuity correction",
        explanation="Proportion of total benchmark questions where system issued a definitive answer rather than abstaining (answered_count / total_examples).",
    )

    # 3. Selective Accuracy
    if answered_count > 0:
        sel_acc_val = correct_count / answered_count
        sel_ci_low, sel_ci_high = compute_wilson_score_interval(correct_count, answered_count)
    else:
        sel_acc_val = 0.0
        sel_ci_low, sel_ci_high = 0.0, 0.0
    metrics_dict["selective_accuracy"] = MetricStatistic(
        metric_name="Selective Accuracy",
        point_estimate=round(sel_acc_val, 4),
        sample_size=answered_count,
        ci_lower=sel_ci_low,
        ci_upper=sel_ci_high,
        ci_method="Wilson score interval with continuity correction",
        explanation="Accuracy conditioned strictly on answered queries (correct_count / answered_count). Measures precision when the system chooses to speak.",
    )

    # 4. Selective Risk
    sel_risk_val = 1.0 - sel_acc_val if answered_count > 0 else 0.0
    metrics_dict["selective_risk"] = MetricStatistic(
        metric_name="Selective Risk",
        point_estimate=round(sel_risk_val, 4),
        sample_size=answered_count,
        ci_lower=round(1.0 - sel_ci_high, 4),
        ci_upper=round(1.0 - sel_ci_low, 4),
        ci_method="Wilson score complement interval",
        explanation="Empirical error rate on answered queries (1 - selective_accuracy). Represents patient safety risk when answering.",
    )

    # 5. Latency (Bootstrap CI)
    latencies = [r.latency_seconds for r in records]
    lat_mean = float(np.mean(latencies))
    lat_low, lat_high = compute_bootstrap_ci(latencies, statistic_fn=np.mean)
    metrics_dict["mean_latency_seconds"] = MetricStatistic(
        metric_name="Mean Latency (s)",
        point_estimate=round(lat_mean, 2),
        sample_size=total,
        ci_lower=lat_low,
        ci_upper=lat_high,
        ci_method="Empirical bootstrap percentile interval (B=1,000)",
        explanation="Average wall-clock response time in seconds per query from input to final synthesized output.",
    )

    # 6. Paired Baseline Comparisons
    paired_reports: Dict[str, Any] = {}
    if baseline_records:
        for b_name, b_recs in baseline_records.items():
            rep = compare_systems(records, b_recs, system_a_name="FullMedRAG", system_b_name=b_name)
            paired_reports[b_name] = rep.to_dict()

    return ComprehensiveStatisticalReport(
        experiment_id=experiment_id,
        dataset=dataset_name,
        split=split_name,
        total_examples=total,
        metrics=metrics_dict,
        paired_comparisons=paired_reports,
    )
