"""
Scientific Paired Statistical Significance Engine for Medical QA Benchmarks.
Implements:
1. McNemar's test (with continuity correction and exact binomial fallback for small discordant counts)
   for paired binary correctness.
2. Paired empirical bootstrap resampling for non-normal continuous and selective metrics
   (Accuracy, Selective Accuracy, Macro-F1, Coverage).
3. Statistical limitations documentation and strict guardrails against false significance claims.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .schema import PredictionRecord


@dataclass
class McNemarResult:
    """Results from paired McNemar's test on binary correctness."""
    statistic: float
    p_value: float
    table: Dict[str, int]               # {"both_correct": a, "a_only": b, "b_only": c, "both_wrong": d}
    method: str                         # "exact_binomial" or "edwards_continuity_corrected_chi2"
    is_significant_05: bool
    is_significant_01: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "statistic": round(self.statistic, 4),
            "p_value": round(self.p_value, 5),
            "contingency_table": self.table,
            "method": self.method,
            "is_significant_p05": self.is_significant_05,
            "is_significant_p01": self.is_significant_01,
        }


@dataclass
class BootstrapResult:
    """Results from paired empirical bootstrap resampling."""
    metric_name: str
    observed_diff: float                # Value(System A) - Value(System B)
    ci_lower: float                     # e.g. 2.5th percentile for 95% CI
    ci_upper: float                     # e.g. 97.5th percentile for 95% CI
    p_value: float                      # Two-tailed empirical bootstrap p-value
    confidence_level: float = 0.95
    num_samples: int = 1000
    is_significant: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric_name,
            "observed_diff": round(self.observed_diff, 4),
            "ci_lower": round(self.ci_lower, 4),
            "ci_upper": round(self.ci_upper, 4),
            "confidence_level": self.confidence_level,
            "bootstrap_p_value": round(self.p_value, 5),
            "is_significant": self.is_significant,
            "num_samples": self.num_samples,
        }


@dataclass
class PairedComparisonReport:
    """Comprehensive paired comparative evaluation between two MedRAG configurations."""
    system_a_name: str
    system_b_name: str
    total_aligned_examples: int
    mcnemar: McNemarResult
    accuracy_bootstrap: BootstrapResult
    selective_acc_bootstrap: Optional[BootstrapResult] = None
    coverage_bootstrap: Optional[BootstrapResult] = None
    macro_f1_bootstrap: Optional[BootstrapResult] = None
    limitations_disclaimer: str = (
        "Notice: Statistical significance indicates that the observed difference is unlikely "
        "under the null hypothesis of equal performance on this sample. It does not establish "
        "clinical efficacy or patient safety in real-world clinical practice."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "comparison": f"{self.system_a_name} vs {self.system_b_name}",
            "system_a": self.system_a_name,
            "system_b": self.system_b_name,
            "total_examples": self.total_aligned_examples,
            "mcnemar_test": self.mcnemar.to_dict(),
            "accuracy_bootstrap": self.accuracy_bootstrap.to_dict(),
            "selective_acc_bootstrap": self.selective_acc_bootstrap.to_dict() if self.selective_acc_bootstrap else None,
            "coverage_bootstrap": self.coverage_bootstrap.to_dict() if self.coverage_bootstrap else None,
            "macro_f1_bootstrap": self.macro_f1_bootstrap.to_dict() if self.macro_f1_bootstrap else None,
            "limitations_disclaimer": self.limitations_disclaimer,
        }

    def to_markdown(self) -> str:
        lines = [
            f"### Statistical Significance: **{self.system_a_name}** vs. **{self.system_b_name}**",
            f"- **Sample Size**: {self.total_aligned_examples} aligned question pairs.",
            f"- **McNemar's Test**: $\\chi^2 = {self.mcnemar.statistic:.3f}$ ($p = {self.mcnemar.p_value:.4f}$, {self.mcnemar.method}).",
            f"  - Discordant pairs: {self.system_a_name} won {self.mcnemar.table['a_only']} questions, {self.system_b_name} won {self.mcnemar.table['b_only']} questions.",
            f"  - Statistically significant at $\\alpha=0.05$: **{'YES' if self.mcnemar.is_significant_05 else 'NO'}**.",
            f"- **Accuracy Difference ($\\Delta$)**: {self.accuracy_bootstrap.observed_diff:+.2%} (95% CI: [{self.accuracy_bootstrap.ci_lower:+.2%}, {self.accuracy_bootstrap.ci_upper:+.2%}], bootstrap $p={self.accuracy_bootstrap.p_value:.4f}$).",
        ]
        if self.selective_acc_bootstrap:
            lines.append(
                f"- **Selective Accuracy $\\Delta$**: {self.selective_acc_bootstrap.observed_diff:+.2%} "
                f"(95% CI: [{self.selective_acc_bootstrap.ci_lower:+.2%}, {self.selective_acc_bootstrap.ci_upper:+.2%}], "
                f"$p={self.selective_acc_bootstrap.p_value:.4f}$)."
            )
        if self.coverage_bootstrap:
            lines.append(
                f"- **Coverage $\\Delta$**: {self.coverage_bootstrap.observed_diff:+.2%} "
                f"(95% CI: [{self.coverage_bootstrap.ci_lower:+.2%}, {self.coverage_bootstrap.ci_upper:+.2%}])."
            )
        lines.append(f"> [!NOTE]\n> {self.limitations_disclaimer}")
        return "\n".join(lines)


# ==============================================================================
# 1. McNemar's Test Implementation
# ==============================================================================

def compute_mcnemar_test(
    correct_a: List[bool],
    correct_b: List[bool],
    cc: bool = True,
) -> McNemarResult:
    """
    Computes McNemar's test for paired binary correctness outcomes.
    Uses exact binomial test when discordant pairs (b + c) < 25,
    and Edwards continuity-corrected chi-squared test otherwise.
    """
    if len(correct_a) != len(correct_b):
        raise ValueError("Inputs must have identical length for paired testing.")

    n = len(correct_a)
    if n == 0:
        return McNemarResult(0.0, 1.0, {"both_correct": 0, "a_only": 0, "b_only": 0, "both_wrong": 0}, "empty", False, False)

    a = 0  # both correct
    b = 0  # A correct, B wrong
    c = 0  # A wrong, B correct
    d = 0  # both wrong

    for y_a, y_b in zip(correct_a, correct_b):
        if y_a and y_b:
            a += 1
        elif y_a and not y_b:
            b += 1
        elif not y_a and y_b:
            c += 1
        else:
            d += 1

    table = {"both_correct": a, "a_only": b, "b_only": c, "both_wrong": d}
    discordant = b + c

    if discordant == 0:
        return McNemarResult(
            statistic=0.0,
            p_value=1.0,
            table=table,
            method="no_discordant_pairs",
            is_significant_05=False,
            is_significant_01=False,
        )

    # Fallback to exact two-tailed binomial test for small sample discordant pairs
    if discordant < 25:
        # Two-tailed exact binomial: p = 2 * sum(comb(n, k) * 0.5^n for k in 0..min(b, c))
        min_k = min(b, c)
        p_val = 0.0
        for k in range(min_k + 1):
            p_val += math.comb(discordant, k) * (0.5 ** discordant)
        p_val = min(1.0, 2.0 * p_val)
        stat = float(min_k)
        method = "exact_binomial"
    else:
        # Edwards continuity-corrected chi-squared with 1 degree of freedom
        diff = abs(b - c)
        num = (diff - 1.0 if cc else diff) ** 2
        chi2 = num / discordant
        # Exact analytical p-value for 1 df: p = erfc(sqrt(chi2 / 2))
        p_val = math.erfc(math.sqrt(chi2 / 2.0))
        p_val = min(1.0, max(0.0, p_val))
        stat = chi2
        method = "edwards_continuity_corrected_chi2" if cc else "standard_chi2"

    return McNemarResult(
        statistic=stat,
        p_value=p_val,
        table=table,
        method=method,
        is_significant_05=(p_val < 0.05),
        is_significant_01=(p_val < 0.01),
    )


# ==============================================================================
# 2. Paired Empirical Bootstrap Resampling
# ==============================================================================

def paired_bootstrap_ci(
    pairs: List[Tuple[PredictionRecord, PredictionRecord]],
    metric_fn: Callable[[List[PredictionRecord]], float],
    metric_name: str,
    num_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> BootstrapResult:
    """
    Computes empirical bootstrap confidence interval for paired metric difference
    Delta = Metric(A) - Metric(B).
    Resamples paired indices with replacement to preserve example-level correlation.
    """
    n = len(pairs)
    if n == 0:
        return BootstrapResult(metric_name, 0.0, 0.0, 0.0, 1.0, confidence_level, num_bootstraps, False)

    recs_a = [p[0] for p in pairs]
    recs_b = [p[1] for p in pairs]

    obs_a = metric_fn(recs_a)
    obs_b = metric_fn(recs_b)
    obs_diff = obs_a - obs_b

    rng = random.Random(random_seed)
    diffs: List[float] = []

    for _ in range(num_bootstraps):
        sampled_indices = [rng.randint(0, n - 1) for _ in range(n)]
        sample_a = [recs_a[idx] for idx in sampled_indices]
        sample_b = [recs_b[idx] for idx in sampled_indices]

        diff = metric_fn(sample_a) - metric_fn(sample_b)
        diffs.append(diff)

    diffs.sort()

    # Percentile confidence interval
    alpha = 1.0 - confidence_level
    lower_idx = int(math.floor(alpha / 2.0 * num_bootstraps))
    upper_idx = int(math.ceil((1.0 - alpha / 2.0) * num_bootstraps)) - 1
    lower_idx = max(0, min(lower_idx, num_bootstraps - 1))
    upper_idx = max(0, min(upper_idx, num_bootstraps - 1))

    ci_lower = diffs[lower_idx]
    ci_upper = diffs[upper_idx]

    # Two-tailed empirical p-value testing null hypothesis Delta = 0
    num_le_zero = sum(1 for d in diffs if d <= 0.0)
    num_ge_zero = sum(1 for d in diffs if d >= 0.0)
    p_val = min(1.0, 2.0 * min(num_le_zero, num_ge_zero) / num_bootstraps)

    # Significant if 95% CI does not overlap 0
    is_sig = (ci_lower > 0.0 and ci_upper > 0.0) or (ci_lower < 0.0 and ci_upper < 0.0)

    return BootstrapResult(
        metric_name=metric_name,
        observed_diff=obs_diff,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_val,
        confidence_level=confidence_level,
        num_samples=num_bootstraps,
        is_significant=is_sig,
    )


# ==============================================================================
# 3. High-Level Paired System Comparison
# ==============================================================================

def compare_systems(
    records_a: List[PredictionRecord],
    records_b: List[PredictionRecord],
    system_a_name: str = "System A",
    system_b_name: str = "System B",
    num_bootstraps: int = 1000,
    random_seed: int = 42,
) -> PairedComparisonReport:
    """
    Performs full paired statistical evaluation between two system runs.
    Aligns records strictly by example_id before analysis.
    """
    # 1. Align records by example_id
    b_map = {r.example_id: r for r in records_b}
    aligned_pairs: List[Tuple[PredictionRecord, PredictionRecord]] = []

    for r_a in records_a:
        if r_a.example_id in b_map:
            aligned_pairs.append((r_a, b_map[r_a.example_id]))

    total_aligned = len(aligned_pairs)
    if total_aligned == 0:
        raise ValueError(
            f"No matching example IDs found between '{system_a_name}' and '{system_b_name}'. "
            "Cannot perform paired statistical testing."
        )

    correct_a = [bool(p[0].is_correct and not p[0].abstained) for p in aligned_pairs]
    correct_b = [bool(p[1].is_correct and not p[1].abstained) for p in aligned_pairs]

    # 2. McNemar's Test for binary correctness
    mcnemar_res = compute_mcnemar_test(correct_a, correct_b)

    # 3. Paired Bootstrap for Accuracy
    def _calc_acc(recs: List[PredictionRecord]) -> float:
        if not recs:
            return 0.0
        return sum(1 for r in recs if r.is_correct and not r.abstained) / len(recs)

    acc_boot = paired_bootstrap_ci(
        aligned_pairs,
        _calc_acc,
        metric_name="Accuracy",
        num_bootstraps=num_bootstraps,
        random_seed=random_seed,
    )

    # 4. Paired Bootstrap for Selective Accuracy
    def _calc_sel_acc(recs: List[PredictionRecord]) -> float:
        answered = [r for r in recs if not r.abstained]
        if not answered:
            return 0.0
        return sum(1 for r in answered if r.is_correct) / len(answered)

    sel_acc_boot = paired_bootstrap_ci(
        aligned_pairs,
        _calc_sel_acc,
        metric_name="Selective Accuracy",
        num_bootstraps=num_bootstraps,
        random_seed=random_seed + 1,
    )

    # 5. Paired Bootstrap for Coverage
    def _calc_cov(recs: List[PredictionRecord]) -> float:
        if not recs:
            return 0.0
        return sum(1 for r in recs if not r.abstained) / len(recs)

    cov_boot = paired_bootstrap_ci(
        aligned_pairs,
        _calc_cov,
        metric_name="Coverage",
        num_bootstraps=num_bootstraps,
        random_seed=random_seed + 2,
    )

    return PairedComparisonReport(
        system_a_name=system_a_name,
        system_b_name=system_b_name,
        total_aligned_examples=total_aligned,
        mcnemar=mcnemar_res,
        accuracy_bootstrap=acc_boot,
        selective_acc_bootstrap=sel_acc_boot,
        coverage_bootstrap=cov_boot,
    )
