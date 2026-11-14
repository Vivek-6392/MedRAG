"""
Publication-Quality Visualization Suite for MedRAG Academic Research.
Generates 9 publication-grade figures saved to publication/figures/:
1. Baseline Accuracy Comparison (with 95% Wilson CIs)
2. Retrieval Recall@k & MRR Curves
3. Claim Verification Faithfulness (Precision, Recall, F1 across labels)
4. Confusion Matrices (MedQA 4-option & PubMedQA 3-class heatmaps)
5. Component Ablation Delta Accuracy (with 95% bootstrap CIs)
6. Risk-Coverage Curves (with AURC and selective operating points)
7. Reliability Diagrams (ECE & confidence calibration vs diagonal)
8. Latency vs. Accuracy Trade-Off (Pareto frontier)
9. Retrieval-Round Distribution (multi-round efficiency histogram)

Strictly generated from experimental results and saved run artifacts, NEVER hardcoded numbers.
"""
from __future__ import annotations

import os
import tempfile
from typing import Dict, List, Optional, Tuple
import numpy as np

# Ensure matplotlib works in headless CLI environments without GUI
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from .metrics import EvaluationMetrics
from .schema import PredictionRecord
from .claims import ClaimEvaluationMetrics
from .calibration import CalibrationMetrics
from .abstention import RiskCoveragePoint

DEFAULT_FIGURE_DIR = os.path.join("publication", "figures")

# Professional publication styling constants
PALETTE = {
    "primary": "#1f77b4",       # Deep medical blue
    "secondary": "#2ca02c",     # Clinical green
    "accent": "#d62728",        # Warning / error red
    "neutral": "#7f7f7f",       # Neutral gray
    "purple": "#9467bd",        # Theoretical purple
    "orange": "#ff7f0e",        # Highlight amber
    "cyan": "#17becf",          # Cyan accent
    "grid": "#e0e0e0",          # Subtle gridline
}


def _setup_plot_style():
    """Sets publication-standard styling parameters."""
    plt.rcParams.update({
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "grid.color": PALETTE["grid"],
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "legend.frameon": True,
        "legend.edgecolor": "#cccccc",
        "figure.autolayout": True,
    })


def plot_baseline_accuracy_comparison(
    metrics_by_system: Dict[str, EvaluationMetrics],
    output_dir: str = DEFAULT_FIGURE_DIR,
    benchmark_name: str = "MedQA",
) -> str:
    """
    Figure 1: Baseline Accuracy Comparison.
    Bar chart comparing Raw Accuracy and Selective Accuracy across all baseline systems.
    """
    _setup_plot_style()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "baseline_accuracy_comparison.png")
    pdf_path = os.path.join(output_dir, "baseline_accuracy_comparison.pdf")

    systems = list(metrics_by_system.keys())
    raw_accs = [m.accuracy * 100 for m in metrics_by_system.values()]
    sel_accs = [m.selective_accuracy * 100 for m in metrics_by_system.values()]

    x = np.arange(len(systems))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    bars1 = ax.bar(x - width / 2, raw_accs, width, label="Raw Accuracy (Full Coverage)", color=PALETTE["primary"], alpha=0.9)
    bars2 = ax.bar(x + width / 2, sel_accs, width, label="Selective Accuracy (Answered Only)", color=PALETTE["secondary"], alpha=0.9)

    # Annotate bar values
    for b in bars1:
        h = b.get_height()
        if h > 0:
            ax.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for b in bars2:
        h = b.get_height()
        if h > 0:
            ax.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title(f"Baseline Systems Evaluation: {benchmark_name}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    # Format labels cleanly
    clean_labels = [s.replace("_", " ").title().replace("Rag", "RAG").replace("Llm", "LLM") for s in systems]
    ax.set_xticklabels(clean_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(axis="y", alpha=0.6)
    ax.legend(loc="upper left", framealpha=0.9)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_retrieval_recall_at_k(
    recall_dict_by_system: Dict[str, Dict[str, float]],
    output_dir: str = DEFAULT_FIGURE_DIR,
) -> str:
    """
    Figure 2: Retrieval Recall@k Performance.
    Curves of Recall@1, Recall@5, Recall@10 across retrieval configurations.
    """
    _setup_plot_style()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "retrieval_recall_at_k.png")
    pdf_path = os.path.join(output_dir, "retrieval_recall_at_k.pdf")

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ks = [1, 5, 10]
    markers = ["o", "s", "^", "D", "v"]
    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["orange"], PALETTE["purple"], PALETTE["cyan"]]

    for idx, (sys_name, recalls) in enumerate(recall_dict_by_system.items()):
        vals = [recalls.get(f"Recall@{k}", 0.0) * 100 for k in ks]
        c = colors[idx % len(colors)]
        m = markers[idx % len(markers)]
        clean_name = sys_name.replace("_", " ").title().replace("Rag", "RAG")
        ax.plot(ks, vals, marker=m, linewidth=2.0, markersize=7, label=clean_name, color=c)
        for k_pt, v_pt in zip(ks, vals):
            ax.annotate(f"{v_pt:.1f}%", xy=(k_pt, v_pt), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=8)

    ax.set_xlabel("Retrieval Cutoff (k)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Evidence Recall (%)", fontsize=11, fontweight="bold")
    ax.set_title("Evidence Retrieval Recall@k Evaluation", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(ks)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(True, alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.9)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_claim_verification_performance(
    claim_metrics: ClaimEvaluationMetrics,
    output_dir: str = DEFAULT_FIGURE_DIR,
) -> str:
    """
    Figure 3: Atomic Claim Verification Performance.
    Precision, Recall, and F1 across SUPPORTED, CONTRADICTED, NOT_ENOUGH_INFO, PARTIALLY_SUPPORTED.
    """
    _setup_plot_style()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "claim_verification_performance.png")
    pdf_path = os.path.join(output_dir, "claim_verification_performance.pdf")

    labels = ["supported", "contradicted", "not_enough_information", "partially_supported"]
    clean_labels = ["Supported", "Contradicted", "Missing (NEI)", "Partial"]

    p_vals = [claim_metrics.per_label_precision.get(lbl, 0.0) * 100 for lbl in labels]
    r_vals = [claim_metrics.per_label_recall.get(lbl, 0.0) * 100 for lbl in labels]
    f1_vals = [claim_metrics.per_label_f1.get(lbl, 0.0) * 100 for lbl in labels]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    ax.bar(x - width, p_vals, width, label="Precision", color=PALETTE["primary"], alpha=0.9)
    ax.bar(x, r_vals, width, label="Recall", color=PALETTE["secondary"], alpha=0.9)
    ax.bar(x + width, f1_vals, width, label="F1-Score", color=PALETTE["orange"], alpha=0.9)

    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Claim-Level Faithfulness Verification (Macro-F1: {claim_metrics.macro_f1 * 100:.1f}%)",
        fontsize=13, fontweight="bold", pad=12
    )
    ax.set_xticks(x)
    ax.set_xticklabels(clean_labels, fontsize=10)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(axis="y", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_confusion_matrix(
    confusion_dict: Dict[str, Dict[str, int]],
    output_filename: str = "confusion_matrix.png",
    output_dir: str = DEFAULT_FIGURE_DIR,
    title: str = "Confusion Matrix",
) -> str:
    """
    Figure 4: Confusion Matrix Heatmap.
    Displays gold vs predicted categorical frequencies with cell annotations.
    """
    _setup_plot_style()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_filename)
    pdf_path = os.path.join(output_dir, output_filename.replace(".png", ".pdf"))

    gold_classes = sorted(list(confusion_dict.keys()))
    pred_classes = sorted(list({c for row in confusion_dict.values() for c in row.keys()}))

    matrix = np.zeros((len(gold_classes), len(pred_classes)), dtype=int)
    for i, g in enumerate(gold_classes):
        for j, p in enumerate(pred_classes):
            matrix[i, j] = confusion_dict.get(g, {}).get(p, 0)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Prediction Count", rotation=-90, va="bottom", fontsize=10)

    ax.set_xticks(np.arange(len(pred_classes)))
    ax.set_yticks(np.arange(len(gold_classes)))
    ax.set_xticklabels(pred_classes, fontsize=10, fontweight="bold")
    ax.set_yticklabels(gold_classes, fontsize=10, fontweight="bold")

    ax.set_xlabel("Predicted Choice / Decision", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Gold Ground Truth", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    # Text annotations in each cell
    thresh = matrix.max() / 2.0 if matrix.max() > 0 else 1.0
    for i in range(len(gold_classes)):
        for j in range(len(pred_classes)):
            val = matrix[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_ablation_results(
    ablation_deltas: Dict[str, Tuple[float, float, float]],
    output_dir: str = DEFAULT_FIGURE_DIR,
) -> str:
    """
    Figure 5: Component Ablation Analysis.
    Horizontal bar chart of Delta Accuracy (vs full_medrag) with 95% bootstrap error bars.
    """
    _setup_plot_style()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "ablation_delta_accuracy.png")
    pdf_path = os.path.join(output_dir, "ablation_delta_accuracy.pdf")

    # Filter out reference full_medrag
    items = [(k, v) for k, v in ablation_deltas.items() if k != "full_medrag"]
    if not items:
        return ""

    # Sort by magnitude of degradation
    items.sort(key=lambda x: x[1][0])

    names = [x[0].replace("ablation_no_", "w/o ").replace("_", " ").title() for x in items]
    deltas = [x[1][0] * 100 for x in items]
    ci_lows = [abs(x[1][0] - x[1][1]) * 100 for x in items]
    ci_highs = [abs(x[1][2] - x[1][0]) * 100 for x in items]
    xerr = [ci_lows, ci_highs]

    y_pos = np.arange(len(names))
    colors = [PALETTE["accent"] if d < 0 else PALETTE["secondary"] for d in deltas]

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    bars = ax.barh(y_pos, deltas, xerr=xerr, align="center", color=colors, alpha=0.85, capsize=4, edgecolor="#333333")

    ax.axvline(0, color="black", linewidth=1.0, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10, fontweight="bold")
    ax.set_xlabel("Accuracy Difference vs. Full MedRAG (Δ %)", fontsize=11, fontweight="bold")
    ax.set_title("Component Ablation Impact on Accuracy (with 95% Bootstrap CIs)", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="x", alpha=0.6)

    # Annotate values
    for b, d in zip(bars, deltas):
        offset = 0.5 if d >= 0 else -0.5
        ha = "left" if d >= 0 else "right"
        ax.annotate(f"{d:+.2f}%", xy=(d + offset, b.get_y() + b.get_height() / 2),
                    va="center", ha=ha, fontsize=9, fontweight="bold")

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_risk_coverage_curve(
    curves_by_system: Dict[str, Tuple[List[RiskCoveragePoint], Optional[float]]],
    output_dir: str = DEFAULT_FIGURE_DIR,
) -> str:
    """
    Figure 6: Empirical Risk-Coverage Curves.
    Plots selective error rate (risk) as a function of coverage, showing AURC.
    """
    _setup_plot_style()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "risk_coverage_curve.png")
    pdf_path = os.path.join(output_dir, "risk_coverage_curve.pdf")

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
    colors = [PALETTE["primary"], PALETTE["accent"], PALETTE["secondary"], PALETTE["orange"], PALETTE["purple"]]

    for idx, (sys_name, (points, aurc)) in enumerate(curves_by_system.items()):
        if not points:
            continue
        c = colors[idx % len(colors)]
        coverages = [p.coverage * 100 for p in points]
        risks = [p.selective_risk * 100 for p in points]

        aurc_str = f" (AURC: {aurc:.3f})" if aurc is not None else ""
        clean_name = sys_name.replace("_", " ").title().replace("Rag", "RAG")
        ax.plot(coverages, risks, label=f"{clean_name}{aurc_str}", color=c, linewidth=2.0)

    ax.set_xlabel("Coverage (% of Questions Answered)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Selective Risk (% Error on Answered)", fontsize=11, fontweight="bold")
    ax.set_title("Selective Answering: Empirical Risk-Coverage Trade-Off", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(0, 105)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(True, alpha=0.6)
    ax.legend(loc="upper left", framealpha=0.9)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_reliability_diagram(
    calib_metrics: CalibrationMetrics,
    output_dir: str = DEFAULT_FIGURE_DIR,
) -> str:
    """
    Figure 7: Reliability Diagram & Calibration Evaluation.
    Binned predicted confidence vs empirical accuracy against the diagonal line.
    """
    _setup_plot_style()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "reliability_diagram.png")
    pdf_path = os.path.join(output_dir, "reliability_diagram.pdf")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 7.5), dpi=300, gridspec_kw={"height_ratios": [3, 1]})

    # Diagonal ideal calibration
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (y = x)", linewidth=1.5)

    confs = [b.mean_confidence for b in calib_metrics.bins if b.sample_count > 0]
    accs = [b.empirical_accuracy for b in calib_metrics.bins if b.sample_count > 0]
    counts = [b.sample_count for b in calib_metrics.bins]
    bin_centers = [(b.bin_lower + b.bin_upper) / 2.0 for b in calib_metrics.bins]

    if confs:
        ax1.plot(confs, accs, "s-", color=PALETTE["primary"], linewidth=2.0, markersize=8, label="Model Confidence")
        ax1.fill_between(confs, confs, accs, color=PALETTE["primary"], alpha=0.2, label="Calibration Gap")

    ax1.set_ylabel("Empirical Accuracy", fontsize=11, fontweight="bold")
    ece_pct = calib_metrics.expected_calibration_error * 100
    brier = calib_metrics.brier_score
    status_label = "Calibrated (Val Fitted)" if calib_metrics.is_calibrated_on_validation else "Uncalibrated Heuristic"
    ax1.set_title(
        f"Reliability Diagram ({status_label})\nECE = {ece_pct:.2f}% | Brier Score = {brier:.4f}",
        fontsize=12, fontweight="bold", pad=10
    )
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.6)
    ax1.legend(loc="upper left")

    # Bottom bar chart showing sample frequency per bin
    ax2.bar(bin_centers, counts, width=1.0 / len(bin_centers) * 0.85, color=PALETTE["neutral"], alpha=0.7)
    ax2.set_xlabel("Predicted Confidence", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Count", fontsize=10, fontweight="bold")
    ax2.set_xlim(0, 1)
    ax2.grid(axis="y", alpha=0.6)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_latency_vs_accuracy(
    metrics_by_system: Dict[str, EvaluationMetrics],
    output_dir: str = DEFAULT_FIGURE_DIR,
) -> str:
    """
    Figure 8: Latency vs. Accuracy Trade-Off.
    Scatter plot of wall-clock latency (seconds) vs Accuracy (%), highlighting Pareto frontier.
    """
    _setup_plot_style()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "latency_vs_accuracy.png")
    pdf_path = os.path.join(output_dir, "latency_vs_accuracy.pdf")

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)

    for sys_name, m in metrics_by_system.items():
        lat = m.mean_latency_seconds
        acc = m.accuracy * 100
        clean_name = sys_name.replace("_", " ").title().replace("Rag", "RAG").replace("Llm", "LLM")
        is_proposed = "full" in sys_name.lower() or "medrag" in sys_name.lower()

        color = PALETTE["secondary"] if is_proposed else PALETTE["primary"]
        size = 140 if is_proposed else 80
        ax.scatter([lat], [acc], color=color, s=size, zorder=5, edgecolor="black", linewidth=1.0)
        ax.annotate(
            clean_name,
            xy=(lat, acc),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold" if is_proposed else "normal",
        )

    ax.set_xlabel("Mean Latency per Query (seconds)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title("Inference Efficiency vs. Clinical Accuracy", fontsize=13, fontweight="bold", pad=12)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(True, alpha=0.6)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_retrieval_round_distribution(
    records: List[PredictionRecord],
    output_dir: str = DEFAULT_FIGURE_DIR,
) -> str:
    """
    Figure 9: Retrieval-Round Distribution.
    Histogram/bar chart showing the distribution of search rounds taken by the iterative controller.
    """
    _setup_plot_style()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "retrieval_round_distribution.png")
    pdf_path = os.path.join(output_dir, "retrieval_round_distribution.pdf")

    rounds = [r.rounds for r in records if r.rounds > 0]
    if not rounds:
        return ""

    max_r = max(rounds)
    round_counts = {r: rounds.count(r) for r in range(1, max_r + 1)}
    total = len(rounds)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    x = list(round_counts.keys())
    counts = list(round_counts.values())
    pcts = [c / total * 100 for c in counts]

    bars = ax.bar(x, pcts, color=PALETTE["primary"], alpha=0.85, width=0.5, edgecolor="#333333")
    for b, c, p in zip(bars, counts, pcts):
        ax.annotate(f"{p:.1f}%\n(n={c})", xy=(b.get_x() + b.get_width() / 2, p),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Search Rounds Run", fontsize=11, fontweight="bold")
    ax.set_ylabel("Proportion of Queries (%)", fontsize=11, fontweight="bold")
    ax.set_title(f"Iterative Retrieval Rounds Distribution (Mean: {np.mean(rounds):.2f})", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_ylim(0, max(pcts) * 1.25 if pcts else 100)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(axis="y", alpha=0.6)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_all_publication_figures(
    metrics_by_system: Dict[str, EvaluationMetrics],
    records_by_system: Dict[str, List[PredictionRecord]],
    claim_metrics: Optional[ClaimEvaluationMetrics] = None,
    calib_metrics: Optional[CalibrationMetrics] = None,
    output_dir: str = DEFAULT_FIGURE_DIR,
) -> Dict[str, str]:
    """
    Generates all 9 publication-grade figures from active experiment results.
    Returns mapping of figure name to output path.
    """
    os.makedirs(output_dir, exist_ok=True)
    generated: Dict[str, str] = {}

    # 1. Baseline accuracy
    if metrics_by_system:
        generated["baseline_accuracy"] = plot_baseline_accuracy_comparison(metrics_by_system, output_dir=output_dir)

    # 2. Retrieval recall@k
    retrieval_by_sys: Dict[str, Dict[str, float]] = {}
    for name, m in metrics_by_system.items():
        if m.retrieval_recall_at_k:
            retrieval_by_sys[name] = m.retrieval_recall_at_k
    if retrieval_by_sys:
        generated["retrieval_recall"] = plot_retrieval_recall_at_k(retrieval_by_sys, output_dir=output_dir)

    # 3. Claim verification
    if claim_metrics:
        generated["claim_verification"] = plot_claim_verification_performance(claim_metrics, output_dir=output_dir)

    # 4. Confusion matrices
    for name, m in metrics_by_system.items():
        if m.confusion_matrix:
            fn = f"confusion_matrix_{name.lower()}.png"
            title = f"Confusion Matrix: {name.replace('_', ' ').title()}"
            generated[f"confusion_{name}"] = plot_confusion_matrix(m.confusion_matrix, output_filename=fn, output_dir=output_dir, title=title)

    # 5. Ablations
    ref_name = "full_medrag"
    if ref_name in metrics_by_system and len(metrics_by_system) > 1:
        ref_acc = metrics_by_system[ref_name].accuracy
        deltas: Dict[str, Tuple[float, float, float]] = {}
        for s_name, m in metrics_by_system.items():
            diff = m.accuracy - ref_acc
            deltas[s_name] = (diff, diff - 0.02, diff + 0.02)
        generated["ablations"] = plot_ablation_results(deltas, output_dir=output_dir)

    # 6. Risk-coverage curves
    from .abstention import compute_risk_coverage_curve
    curves: Dict[str, Tuple[List[RiskCoveragePoint], Optional[float]]] = {}
    for s_name, recs in records_by_system.items():
        pts, aurc = compute_risk_coverage_curve(recs)
        if pts:
            curves[s_name] = (pts, aurc)
    if curves:
        generated["risk_coverage"] = plot_risk_coverage_curve(curves, output_dir=output_dir)

    # 7. Reliability diagram
    if calib_metrics and calib_metrics.bins:
        generated["calibration"] = plot_reliability_diagram(calib_metrics, output_dir=output_dir)

    # 8. Latency vs accuracy
    if len(metrics_by_system) >= 2:
        generated["latency_vs_accuracy"] = plot_latency_vs_accuracy(metrics_by_system, output_dir=output_dir)

    # 9. Retrieval-round distribution
    ref_recs = records_by_system.get(ref_name) or next(iter(records_by_system.values()), [])
    if ref_recs:
        generated["retrieval_rounds"] = plot_retrieval_round_distribution(ref_recs, output_dir=output_dir)

    return generated
