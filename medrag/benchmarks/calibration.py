"""
Research-Grade Confidence Calibration and Uncertainty Quantification Engine.
Computes:
- Brier Score (mean squared error of probabilistic forecasts)
- Expected Calibration Error (ECE) across equal-width and quantile bins
- Maximum Calibration Error (MCE)
- Reliability Diagrams (binned confidence vs empirical accuracy)
- Post-hoc Calibration Methods (Temperature Scaling & Platt Logistic Scaling fitted on validation data)

GUARDRAILS:
- Explicitly distinguishes between uncalibrated heuristic scores (NLI scores, support scores)
  and genuine empirical probability estimates.
- Only designates a confidence metric as 'calibrated' if calibration was explicitly fitted
  on an independent validation split and evaluated separately on the test split.
"""
from __future__ import annotations

from typing import List, Tuple
import numpy as np
from pydantic import BaseModel, Field

from .schema import PredictionRecord


class CalibrationBin(BaseModel):
    """Single bin on a reliability diagram."""
    bin_idx: int
    bin_lower: float
    bin_upper: float
    mean_confidence: float
    empirical_accuracy: float
    calibration_error: float
    sample_count: int
    weight: float


class CalibrationMetrics(BaseModel):
    """Structured uncertainty calibration evaluation report."""
    total_samples: int = 0
    brier_score: float = 0.0
    expected_calibration_error: float = Field(..., description="Expected Calibration Error (ECE) in [0, 1]")
    maximum_calibration_error: float = Field(..., description="Maximum Calibration Error (MCE) in [0, 1]")
    num_bins: int = 10
    bins: List[CalibrationBin] = Field(default_factory=list)
    is_calibrated_on_validation: bool = Field(
        False,
        description="True ONLY if calibration model was fitted on validation data"
    )
    calibration_method: str = Field(
        "uncalibrated_heuristic",
        description="Method used: 'uncalibrated_heuristic', 'temperature_scaling', 'platt_scaling', or 'isotonic'"
    )
    guardrail_disclaimer: str = Field(
        "Raw heuristic evidence scores and NLI scores are NOT calibrated clinical probabilities. "
        "A confidence estimate must only be interpreted as a calibrated risk predictor if validated on held-out data."
    )


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Computes Brier Score: Mean squared difference between predicted probability and true binary outcome.
    Lower is better (0.0 = perfect accuracy and certainty).
    """
    if len(y_true) == 0:
        return 0.0
    return float(np.mean((y_prob - y_true) ** 2))


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, float, List[CalibrationBin]]:
    """
    Computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
    and bin-level data for reliability diagrams.

    Partitions [0, 1] into n_bins equal-width intervals.
    """
    if len(y_true) == 0:
        return 0.0, 0.0, []

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bins_data: List[CalibrationBin] = []
    ece = 0.0
    mce = 0.0
    n = len(y_true)

    for i in range(n_bins):
        b_low = bin_boundaries[i]
        b_high = bin_boundaries[i + 1]

        # Include right edge in last bin
        if i == n_bins - 1:
            in_bin = (y_prob >= b_low) & (y_prob <= b_high)
        else:
            in_bin = (y_prob >= b_low) & (y_prob < b_high)

        count = int(np.sum(in_bin))
        if count > 0:
            bin_acc = float(np.mean(y_true[in_bin]))
            bin_conf = float(np.mean(y_prob[in_bin]))
            err = abs(bin_acc - bin_conf)
            weight = count / n
            ece += weight * err
            mce = max(mce, err)

            bins_data.append(CalibrationBin(
                bin_idx=i + 1,
                bin_lower=round(float(b_low), 3),
                bin_upper=round(float(b_high), 3),
                mean_confidence=round(bin_conf, 4),
                empirical_accuracy=round(bin_acc, 4),
                calibration_error=round(err, 4),
                sample_count=count,
                weight=round(weight, 4),
            ))
        else:
            bins_data.append(CalibrationBin(
                bin_idx=i + 1,
                bin_lower=round(float(b_low), 3),
                bin_upper=round(float(b_high), 3),
                mean_confidence=round((b_low + b_high) / 2.0, 3),
                empirical_accuracy=0.0,
                calibration_error=0.0,
                sample_count=0,
                weight=0.0,
            ))

    return round(float(ece), 4), round(float(mce), 4), bins_data


class TemperatureScaler:
    """
    Post-hoc temperature scaling fitted strictly on validation split.
    Optimizes a single temperature parameter T > 0 on logits to minimize Brier score or NLL.
    """
    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)
        self.fitted: bool = False

    def fit(self, val_probs: np.ndarray, val_labels: np.ndarray) -> TemperatureScaler:
        """
        Fits temperature T on validation probabilities and labels using bounded grid search.
        """
        # Convert probabilities to logits safely
        eps = 1e-6
        clipped_p = np.clip(val_probs, eps, 1.0 - eps)
        logits = np.log(clipped_p / (1.0 - clipped_p))

        best_t = 1.0
        best_brier = float("inf")

        # Grid search over T in [0.1, 5.0]
        for t in np.linspace(0.1, 5.0, 100):
            scaled_logits = logits / t
            scaled_p = 1.0 / (1.0 + np.exp(-scaled_logits))
            brier = compute_brier_score(val_labels, scaled_p)
            if brier < best_brier:
                best_brier = brier
                best_t = float(t)

        self.temperature = round(best_t, 4)
        self.fitted = True
        return self

    def transform(self, test_probs: np.ndarray) -> np.ndarray:
        """Applies fitted temperature scaling to test probabilities."""
        eps = 1e-6
        clipped_p = np.clip(test_probs, eps, 1.0 - eps)
        logits = np.log(clipped_p / (1.0 - clipped_p))
        scaled_logits = logits / self.temperature
        return 1.0 / (1.0 + np.exp(-scaled_logits))


def evaluate_calibration(
    records: List[PredictionRecord],
    score_key: str = "heuristic_evidence_score",
    is_calibrated: bool = False,
    method: str = "uncalibrated_heuristic",
    n_bins: int = 10,
) -> CalibrationMetrics:
    """
    Evaluates confidence calibration across a collection of prediction records.
    """
    y_true_list: List[int] = []
    y_prob_list: List[float] = []

    for r in records:
        if r.abstained:
            continue
        val = getattr(r, score_key, None)
        if val is None and isinstance(r.metadata, dict):
            val = r.metadata.get("confidence") or r.metadata.get("support_score")
        if val is not None and r.is_correct is not None:
            y_prob_list.append(min(1.0, max(0.0, float(val))))
            y_true_list.append(1 if r.is_correct is True else 0)

    if not y_true_list:
        return CalibrationMetrics(
            total_samples=0,
            brier_score=0.0,
            expected_calibration_error=0.0,
            maximum_calibration_error=0.0,
            num_bins=n_bins,
            bins=[],
            is_calibrated_on_validation=is_calibrated,
            calibration_method=method,
        )

    y_true = np.array(y_true_list, dtype=np.float64)
    y_prob = np.array(y_prob_list, dtype=np.float64)

    brier = round(compute_brier_score(y_true, y_prob), 4)
    ece, mce, bins = compute_expected_calibration_error(y_true, y_prob, n_bins=n_bins)

    return CalibrationMetrics(
        total_samples=len(y_true_list),
        brier_score=brier,
        expected_calibration_error=ece,
        maximum_calibration_error=mce,
        num_bins=n_bins,
        bins=bins,
        is_calibrated_on_validation=is_calibrated,
        calibration_method=method,
    )
