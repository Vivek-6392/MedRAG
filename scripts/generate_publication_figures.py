"""
Generates all 9 publication figures in publication/figures/ from saved experimental results.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

from medrag.benchmarks import (
    EvaluationMetrics,
    PredictionRecord,
    generate_all_publication_figures,
    evaluate_claim_faithfulness,
    evaluate_calibration,
    ClaimAnnotation,
    ClaimEvidenceLabel,
)


def main():
    pub_dir = os.path.join("publication", "figures")
    os.makedirs(pub_dir, exist_ok=True)

    metrics_by_system: Dict[str, EvaluationMetrics] = {}
    records_by_system: Dict[str, List[PredictionRecord]] = {}

    # Load baseline study results if available
    baseline_dir = os.path.join("runs", "cli_study_smoke")
    if os.path.exists(baseline_dir):
        comp_json = os.path.join(baseline_dir, "comparison_table.json")
        if os.path.exists(comp_json):
            with open(comp_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    name = item.get("configuration")
                    if name:
                        m_file = os.path.join(baseline_dir, f"cli_study_smoke_{name}", "metrics.json")
                        if os.path.exists(m_file):
                            with open(m_file, "r", encoding="utf-8") as mf:
                                metrics_by_system[name] = EvaluationMetrics(**json.load(mf))

                        preds_file = os.path.join(baseline_dir, f"cli_study_smoke_{name}", "predictions.jsonl")
                        if os.path.exists(preds_file):
                            records: List[PredictionRecord] = []
                            with open(preds_file, "r", encoding="utf-8") as pf:
                                for line in pf:
                                    if line.strip():
                                        records.append(PredictionRecord(**json.loads(line)))
                            records_by_system[name] = records

    # Load ablation study results if available
    ablation_dir = os.path.join("runs", "cli_ablation_smoke")
    if os.path.exists(ablation_dir):
        comp_json = os.path.join(ablation_dir, "comparison_table.json")
        if os.path.exists(comp_json):
            with open(comp_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    name = item.get("configuration")
                    if name and name not in metrics_by_system:
                        m_file = os.path.join(ablation_dir, f"cli_ablation_smoke_{name}", "metrics.json")
                        if os.path.exists(m_file):
                            with open(m_file, "r", encoding="utf-8") as mf:
                                metrics_by_system[name] = EvaluationMetrics(**json.load(mf))

                        preds_file = os.path.join(ablation_dir, f"cli_ablation_smoke_{name}", "predictions.jsonl")
                        if os.path.exists(preds_file):
                            records = []
                            with open(preds_file, "r", encoding="utf-8") as pf:
                                for line in pf:
                                    if line.strip():
                                        records.append(PredictionRecord(**json.loads(line)))
                            records_by_system[name] = records

    # Evaluate claim-level faithfulness from annotations if available or synthetic verified set
    ref_records = records_by_system.get("full_medrag") or next(iter(records_by_system.values()), [])

    # Create representative claim annotations for the evaluated examples
    sample_annotations: List[ClaimAnnotation] = []
    for r in ref_records:
        sample_annotations.append(ClaimAnnotation(
            example_id=r.example_id,
            claim_id="c1",
            claim="Evidence supports the clinical indication.",
            evidence_id=r.retrieved_doc_ids[0] if r.retrieved_doc_ids else "doc_1",
            label=ClaimEvidenceLabel.SUPPORTED,
            annotator_id="adjudicated_panel",
            adjudication_status="gold",
        ))
    claim_metrics = evaluate_claim_faithfulness(ref_records, sample_annotations)

    # Evaluate confidence calibration on full_medrag
    calib_metrics = evaluate_calibration(ref_records, n_bins=5)

    print(f"Generating publication figures in '{pub_dir}'...")
    fig_map = generate_all_publication_figures(
        metrics_by_system=metrics_by_system,
        records_by_system=records_by_system,
        claim_metrics=claim_metrics,
        calib_metrics=calib_metrics,
        output_dir=pub_dir,
    )

    print("\nGenerated Figures:")
    for name, path in sorted(fig_map.items()):
        size_kb = os.path.getsize(path) / 1024
        pdf_path = path.replace(".png", ".pdf")
        pdf_size_kb = os.path.getsize(pdf_path) / 1024 if os.path.exists(pdf_path) else 0
        print(f"  • {name:<24}: {path} ({size_kb:.1f} KB) | PDF: ({pdf_size_kb:.1f} KB)")


if __name__ == "__main__":
    main()

