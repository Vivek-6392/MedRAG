"""
Multi-Configuration Study Execution Engine for MedRAG.
Coordinates execution of standardized baseline and ablation suites,
enforces fairness constraints, produces combined comparison and ablation tables,
exports per-example prediction comparisons (CSV/JSONL), and computes paired
statistical significance reports.
"""
from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..config_schema import ExperimentConfig
from ..logger import get_logger
from ..retriever import Document
from .experiments import validate_fairness
from .metrics import EvaluationMetrics
from .runner import EvaluationRunner
from .schema import PredictionRecord
from .significance import PairedComparisonReport, compare_systems
from .statistical_report import generate_comprehensive_statistical_report
from .visualizations import generate_all_publication_figures

logger = get_logger("medrag.benchmarks.study")


@dataclass
class StudyResult:
    """Comprehensive aggregated result of a multi-configuration benchmark study."""
    study_id: str
    benchmark_name: str
    split: str
    configs: List[ExperimentConfig]
    metrics: Dict[str, EvaluationMetrics]
    comparison_table_md: str
    ablation_table_md: Optional[str] = None
    significance_reports: List[PairedComparisonReport] = field(default_factory=list)
    output_dir: str = ""

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "study_id": self.study_id,
            "benchmark_name": self.benchmark_name,
            "split": self.split,
            "num_configurations": len(self.configs),
            "metrics": {name: m.model_dump() for name, m in self.metrics.items()},
            "significance_comparisons": [r.to_dict() for r in self.significance_reports],
        }


class StudyRunner:
    """
    Executes multiple benchmark configurations in one unified workflow.
    Guarantees experimental fairness across dataset splits, models, and budgets.
    """

    def __init__(
        self,
        study_id: str,
        configs: List[ExperimentConfig],
        pipeline_factory: Optional[Any] = None,
        corpus: Optional[List[Document]] = None,
        output_dir: str = "runs",
    ) -> None:
        self.study_id = study_id
        self.configs = configs
        self.pipeline_factory = pipeline_factory
        self.corpus = corpus
        self.output_dir = os.path.join(output_dir, study_id)
        os.makedirs(self.output_dir, exist_ok=True)

        # Enforce Part C Fairness
        is_fair, violations = validate_fairness(self.configs)
        if not is_fair:
            warn_msg = f"Experimental fairness violations detected in study '{study_id}':\n" + "\n".join(violations)
            logger.warning(warn_msg)

    def run(self) -> StudyResult:
        """
        Executes all configured experiments sequentially, records per-example
        predictions, computes comparative metrics, ablation deltas, and statistical tests.
        """
        logger.info(
            f"Starting Benchmark Study '{self.study_id}' with {len(self.configs)} configurations."
        )
        t_start = time.time()
        metrics_by_config: Dict[str, EvaluationMetrics] = {}
        records_by_config: Dict[str, List[PredictionRecord]] = {}

        for idx, cfg in enumerate(self.configs, 1):
            logger.info(
                f"\n[{idx}/{len(self.configs)}] Running Configuration: '{cfg.protocol}' "
                f"(experiment_id='{cfg.experiment_id}')"
            )

            # Ensure run writes inside this study's directory
            cfg_copy = cfg.model_copy(deep=True)
            cfg_copy.output_dir = self.output_dir

            pipeline = self.pipeline_factory(cfg_copy) if self.pipeline_factory else None
            runner = EvaluationRunner(
                config=cfg_copy,
                pipeline=pipeline,
                corpus=self.corpus,
            )

            metrics = runner.run()
            metrics_by_config[cfg.protocol] = metrics

            all_records = runner.recorder.load_all_predictions()
            records_by_config[cfg.protocol] = all_records

        # 1. Compile Combined Comparison Table
        comparison_md, comparison_json = self._build_comparison_table(metrics_by_config)
        with open(os.path.join(self.output_dir, "comparison_table.md"), "w", encoding="utf-8") as f:
            f.write(comparison_md)
        with open(os.path.join(self.output_dir, "comparison_table.json"), "w", encoding="utf-8") as f:
            json.dump(comparison_json, f, indent=2)

        # 2. Compile Ablation Table (if full_medrag is present)
        ablation_md: Optional[str] = None
        sig_reports: List[PairedComparisonReport] = []
        ref_key = "full_medrag" if "full_medrag" in records_by_config else None

        if ref_key and len(records_by_config) > 1:
            ref_records = records_by_config[ref_key]
            for name, recs in records_by_config.items():
                if name == ref_key:
                    continue
                try:
                    rep = compare_systems(
                        records_a=ref_records,
                        records_b=recs,
                        system_a_name=ref_key,
                        system_b_name=name,
                    )
                    sig_reports.append(rep)
                except Exception as e:
                    logger.warning(f"Could not compute statistical significance for {name}: {e}")

            ablation_md, ablation_json = self._build_ablation_table(
                ref_key=ref_key,
                metrics_by_config=metrics_by_config,
                sig_reports=sig_reports,
            )
            with open(os.path.join(self.output_dir, "ablation_table.md"), "w", encoding="utf-8") as f:
                f.write(ablation_md)
            with open(os.path.join(self.output_dir, "ablation_table.json"), "w", encoding="utf-8") as f:
                json.dump(ablation_json, f, indent=2)

        # 3. Export Per-Example Prediction Comparisons (CSV and JSONL)
        self._export_per_example_comparisons(records_by_config)

        # 4. Export Statistical Significance Report & Comprehensive Statistics
        sig_dicts = [r.to_dict() for r in sig_reports]
        with open(os.path.join(self.output_dir, "statistical_report.json"), "w", encoding="utf-8") as f:
            json.dump(sig_dicts, f, indent=2)

        # Generate Comprehensive Markdown Statistical Report for Primary Reference System
        ref_recs = records_by_config.get(ref_key) or next(iter(records_by_config.values()), [])
        baseline_pool = {k: v for k, v in records_by_config.items() if k != ref_key}
        comp_stat_report = generate_comprehensive_statistical_report(
            records=ref_recs,
            baseline_records=baseline_pool if baseline_pool else None,
            experiment_id=self.study_id,
        )
        with open(os.path.join(self.output_dir, "statistical_report.md"), "w", encoding="utf-8") as f:
            f.write(comp_stat_report.to_markdown())

        # 5. Generate Publication-Quality Visualizations in publication/figures and study dir
        try:
            generate_all_publication_figures(
                metrics_by_system=metrics_by_config,
                records_by_system=records_by_config,
                output_dir=os.path.join("publication", "figures"),
            )
            generate_all_publication_figures(
                metrics_by_system=metrics_by_config,
                records_by_system=records_by_config,
                output_dir=os.path.join(self.output_dir, "figures"),
            )
        except Exception as e:
            logger.warning(f"Could not generate some publication figures: {e}")

        total_time = round(time.time() - t_start, 2)
        logger.info(f"\nStudy '{self.study_id}' finished in {total_time}s across {len(self.configs)} configs.")

        return StudyResult(
            study_id=self.study_id,
            benchmark_name=self.configs[0].benchmark_name if self.configs else "",
            split=self.configs[0].split if self.configs else "",
            configs=self.configs,
            metrics=metrics_by_config,
            comparison_table_md=comparison_md,
            ablation_table_md=ablation_md,
            significance_reports=sig_reports,
            output_dir=self.output_dir,
        )

    def _build_comparison_table(
        self,
        metrics_by_config: Dict[str, EvaluationMetrics],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Builds formatted Markdown and JSON comparison table across all configs."""
        headers = [
            "Configuration", "Total", "Raw Acc (%)", "Selective Acc (%)",
            "Selective Risk", "Macro-F1", "Abstention (%)", "Coverage (%)",
            "Mean Rounds", "Latency (s)",
        ]
        rows: List[List[str]] = []
        table_dicts: List[Dict[str, Any]] = []

        for name, m in metrics_by_config.items():
            row = [
                name,
                str(m.total_examples),
                f"{m.accuracy * 100:.1f}%",
                f"{m.selective_accuracy * 100:.1f}%",
                f"{m.selective_risk:.3f}",
                f"{m.macro_f1:.3f}" if m.macro_f1 is not None else "-",
                f"{m.abstention_rate * 100:.1f}%",
                f"{m.coverage * 100:.1f}%",
                f"{m.mean_retrieval_rounds:.1f}",
                f"{m.mean_latency_seconds:.2f}s",
            ]
            rows.append(row)
            table_dicts.append({
                "configuration": name,
                "total_examples": m.total_examples,
                "accuracy": m.accuracy,
                "selective_accuracy": m.selective_accuracy,
                "selective_risk": m.selective_risk,
                "macro_f1": m.macro_f1,
                "abstention_rate": m.abstention_rate,
                "coverage": m.coverage,
                "mean_rounds": m.mean_retrieval_rounds,
                "mean_latency": m.mean_latency_seconds,
            })

        # Generate markdown string
        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join([":---" if i == 0 else "---:" for i in range(len(headers))]) + " |"
        data_lines = ["| " + " | ".join(r) + " |" for r in rows]
        md_text = "\n".join([f"## Benchmark Comparison Table: `{self.study_id}`", "", header_line, sep_line] + data_lines)
        return md_text, table_dicts

    def _build_ablation_table(
        self,
        ref_key: str,
        metrics_by_config: Dict[str, EvaluationMetrics],
        sig_reports: List[PairedComparisonReport],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Builds leave-one-out ablation table with deltas and statistical confidence intervals."""
        ref_m = metrics_by_config[ref_key]
        sig_map = {r.system_b_name: r for r in sig_reports}

        headers = [
            "Configuration", "Raw Acc (%)", "Δ Acc", "95% Bootstrap CI",
            "McNemar p", "Selective Acc (%)", "Coverage (%)", "Significant?",
        ]
        rows: List[List[str]] = []
        table_dicts: List[Dict[str, Any]] = []

        # Reference row
        rows.append([
            f"**{ref_key} (Reference)**",
            f"{ref_m.accuracy * 100:.1f}%",
            "-",
            "-",
            "-",
            f"{ref_m.selective_accuracy * 100:.1f}%",
            f"{ref_m.coverage * 100:.1f}%",
            "-",
        ])

        for name, m in metrics_by_config.items():
            if name == ref_key:
                continue

            delta_acc = m.accuracy - ref_m.accuracy
            sig_info = sig_map.get(name)

            ci_str = "-"
            p_val_str = "-"
            is_sig_str = "No"

            if sig_info:
                ci = sig_info.accuracy_bootstrap
                ci_str = f"[{ci.ci_lower:+.2%}, {ci.ci_upper:+.2%}]"
                p_val_str = f"{sig_info.mcnemar.p_value:.4f}"
                is_sig_str = "**YES** (p<0.05)" if sig_info.mcnemar.is_significant_05 else "No"

            rows.append([
                name,
                f"{m.accuracy * 100:.1f}%",
                f"{delta_acc:+.2%}",
                ci_str,
                p_val_str,
                f"{m.selective_accuracy * 100:.1f}%",
                f"{m.coverage * 100:.1f}%",
                is_sig_str,
            ])
            table_dicts.append({
                "configuration": name,
                "accuracy": m.accuracy,
                "delta_accuracy": delta_acc,
                "ci_lower": sig_info.accuracy_bootstrap.ci_lower if sig_info else None,
                "ci_upper": sig_info.accuracy_bootstrap.ci_upper if sig_info else None,
                "mcnemar_p_value": sig_info.mcnemar.p_value if sig_info else None,
                "is_significant": sig_info.mcnemar.is_significant_05 if sig_info else False,
            })

        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join([":---" if i == 0 else "---:" for i in range(len(headers))]) + " |"
        data_lines = ["| " + " | ".join(r) + " |" for r in rows]
        md_text = "\n".join([f"## Component Ablation Analysis: vs. `{ref_key}`", "", header_line, sep_line] + data_lines)
        return md_text, table_dicts

    def _export_per_example_comparisons(
        self,
        records_by_config: Dict[str, List[PredictionRecord]],
    ) -> None:
        """Exports side-by-side example predictions across all configurations."""
        if not records_by_config:
            return

        first_config = list(records_by_config.keys())[0]
        example_order = [r.example_id for r in records_by_config[first_config]]

        # Build map: example_id -> record for each config
        maps = {cfg: {r.example_id: r for r in recs} for cfg, recs in records_by_config.items()}

        csv_rows: List[Dict[str, Any]] = []
        jsonl_rows: List[Dict[str, Any]] = []

        config_names = list(records_by_config.keys())

        for eid in example_order:
            ref_rec = maps[first_config].get(eid)
            if not ref_rec:
                continue

            entry: Dict[str, Any] = {
                "example_id": eid,
                "question": ref_rec.question,
                "gold_answer": ref_rec.gold_answer,
            }

            for cname in config_names:
                rec = maps[cname].get(eid)
                if rec:
                    entry[f"{cname}_pred"] = rec.normalized_answer or "ABSTAIN"
                    entry[f"{cname}_correct"] = bool(rec.is_correct and not rec.abstained)
                    entry[f"{cname}_abstained"] = rec.abstained
                else:
                    entry[f"{cname}_pred"] = "MISSING"
                    entry[f"{cname}_correct"] = False
                    entry[f"{cname}_abstained"] = True

            csv_rows.append(entry)
            jsonl_rows.append(entry)

        # Write CSV
        csv_path = os.path.join(self.output_dir, "per_example_comparison.csv")
        if csv_rows:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
                writer.writeheader()
                writer.writerows(csv_rows)

        # Write JSONL
        jsonl_path = os.path.join(self.output_dir, "per_example_comparison.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for row in jsonl_rows:
                f.write(json.dumps(row) + "\n")
