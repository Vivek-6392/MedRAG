"""Inspection and validation script for user's local dataset directories:
- PubMedQA: C:\\Users\\vivek\\Downloads\\pubmedqa-master
- MedQA / data_clean: C:\\Users\\vivek\\Downloads\\data_clean

Performs deep structural audit, schema verification, label distribution checks,
and tests live loading through MedRAG benchmark adapters without modifying,
moving, or fabricating any files.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add workspace to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from medrag.benchmarks.adapters.medqa import MedQAAdapter
from medrag.benchmarks.adapters.pubmedqa import PubMedQAAdapter

PUBMEDQA_DIR = Path(r"C:\Users\vivek\Downloads\pubmedqa-master")
DATA_CLEAN_DIR = Path(r"C:\Users\vivek\Downloads\data_clean")


def inspect_pubmedqa(root_dir: Path) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "searched_root": str(root_dir),
        "exists": root_dir.exists(),
        "files_found": [],
        "pqal_found": False,
        "pqal_path": None,
        "test_gt_found": False,
        "test_gt_path": None,
        "generated_splits_found": [],
        "total_pqal_instances": 0,
        "label_distribution": {},
        "schema_valid": False,
        "test_gt_count": 0,
        "test_overlap": 0,
    }

    if not root_dir.exists():
        return report

    for p in root_dir.rglob("*"):
        if p.is_file():
            report["files_found"].append(str(p.relative_to(root_dir)))

    # Locate ori_pqal.json
    pqal_candidates = list(root_dir.rglob("ori_pqal.json"))
    if pqal_candidates:
        report["pqal_found"] = True
        report["pqal_path"] = str(pqal_candidates[0])

    # Locate test_ground_truth.json
    gt_candidates = list(root_dir.rglob("test_ground_truth.json"))
    if gt_candidates:
        report["test_gt_found"] = True
        report["test_gt_path"] = str(gt_candidates[0])

    # Check for any generated split files (test_set.json, dev_set.json, etc.)
    for s_name in ("test_set.json", "train_set.json", "dev_set.json", "test.json", "train.json", "dev.json"):
        matches = list(root_dir.rglob(s_name))
        for m in matches:
            report["generated_splits_found"].append(str(m.relative_to(root_dir)))

    # Validate ori_pqal.json
    if report["pqal_found"]:
        with open(report["pqal_path"], "r", encoding="utf-8") as f:
            data = json.load(f)
        report["total_pqal_instances"] = len(data)

        labels: Dict[str, int] = {"yes": 0, "no": 0, "maybe": 0}
        invalid_schema = 0
        for pmid, record in data.items():
            if not isinstance(record, dict):
                invalid_schema += 1
                continue
            if "QUESTION" not in record or not record["QUESTION"]:
                invalid_schema += 1
            if "CONTEXTS" not in record or not isinstance(record["CONTEXTS"], list):
                invalid_schema += 1
            dec = str(record.get("final_decision", "")).strip().lower()
            if dec in labels:
                labels[dec] += 1
            else:
                labels[dec] = labels.get(dec, 0) + 1

        report["label_distribution"] = labels
        report["schema_valid"] = (invalid_schema == 0) and (report["total_pqal_instances"] == 1000)

    # Validate test_ground_truth.json
    if report["test_gt_found"]:
        with open(report["test_gt_path"], "r", encoding="utf-8") as f:
            gt_data = json.load(f)
        report["test_gt_count"] = len(gt_data)
        if report["pqal_found"]:
            overlap = [k for k in gt_data if k in data]
            report["test_overlap"] = len(overlap)

    return report


def inspect_medqa(root_dir: Path) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "searched_root": str(root_dir),
        "exists": root_dir.exists(),
        "is_medqa": False,
        "subsets_found": [],
        "us_4options_splits": {},
        "us_5options_splits": {},
        "textbooks_found": [],
        "schema_valid": False,
        "sample_item": None,
    }

    if not root_dir.exists():
        return report

    # Filter out macOS archive metadata (__MACOSX)
    real_paths = [p for p in root_dir.rglob("*") if "__MACOSX" not in p.parts]

    # Check for known MedQA signature directories
    has_questions = any("questions" in p.name.lower() for p in real_paths if p.is_dir())
    has_textbooks = any("textbooks" in p.name.lower() for p in real_paths if p.is_dir())
    if has_questions and has_textbooks:
        report["is_medqa"] = True

    # Identify regional subsets
    for region in ("US", "Mainland", "Taiwan"):
        if any(region in p.parts for p in real_paths):
            report["subsets_found"].append(region)

    # Check US 4_options splits
    four_opt_dir = None
    for p in real_paths:
        if p.is_dir() and p.name == "4_options" and "US" in p.parts:
            four_opt_dir = p
            break

    if four_opt_dir and four_opt_dir.exists():
        for s in ("train", "dev", "test"):
            # Check phrases_no_exclude_{split}.jsonl or {split}.jsonl
            f1 = four_opt_dir / f"phrases_no_exclude_{s}.jsonl"
            f2 = four_opt_dir / f"{s}.jsonl"
            if f1.exists() and f1.stat().st_size > 0:
                report["us_4options_splits"][s] = {"path": str(f1), "size": f1.stat().st_size}
            elif f2.exists() and f2.stat().st_size > 0:
                report["us_4options_splits"][s] = {"path": str(f2), "size": f2.stat().st_size}

    # Check US 5_options splits
    us_dir = None
    for p in real_paths:
        if p.is_dir() and p.name == "US" and (p / "train.jsonl").exists():
            us_dir = p
            break
    if us_dir:
        for s in ("train", "dev", "test"):
            f = us_dir / f"{s}.jsonl"
            if f.exists() and f.stat().st_size > 0:
                report["us_5options_splits"][s] = {"path": str(f), "size": f.stat().st_size}

    # Check English reference textbooks
    tb_en_dir = None
    for p in real_paths:
        if p.is_dir() and p.name == "en" and "textbooks" in p.parts:
            tb_en_dir = p
            break
    if tb_en_dir:
        txt_files = [f for f in tb_en_dir.glob("*.txt") if f.stat().st_size > 0]
        report["textbooks_dir"] = str(tb_en_dir)
        report["textbooks_found"] = [f.name for f in txt_files]
        report["textbooks_total_mb"] = round(sum(f.stat().st_size for f in txt_files) / (1024 * 1024), 2)

    # Validate 4_options schemas & counts
    if len(report["us_4options_splits"]) == 3:
        counts = {}
        opt_dists = {}
        invalid_items = 0
        for s, s_info in report["us_4options_splits"].items():
            cnt = 0
            opt_d = {"A": 0, "B": 0, "C": 0, "D": 0}
            with open(s_info["path"], "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    if cnt == 0 and s == "test":
                        report["sample_item"] = {
                            "question": item.get("question")[:120] + "...",
                            "options": item.get("options"),
                            "answer_idx": item.get("answer_idx"),
                        }
                    q = item.get("question")
                    opts = item.get("options", {})
                    ans = item.get("answer_idx")
                    if not q or not isinstance(opts, dict) or len(opts) != 4 or ans not in ("A", "B", "C", "D"):
                        invalid_items += 1
                    else:
                        opt_d[ans] += 1
                    cnt += 1
            counts[s] = cnt
            opt_dists[s] = opt_d
        report["us_4options_counts"] = counts
        report["us_4options_option_distributions"] = opt_dists
        report["schema_valid"] = (invalid_items == 0)

    return report


def test_adapters_live(medqa_q_dir: str, medqa_tb_dir: str, pubmed_dir: str) -> Dict[str, Any]:
    """Tests that MedRAG's actual benchmark adapters load data from these exact directories."""
    res = {}

    # Test MedQA Adapter
    try:
        medqa_adapter = MedQAAdapter(data_path=medqa_q_dir)
        test_examples = medqa_adapter.load_split("test", limit=3)
        res["medqa_adapter"] = {
            "status": "PASS",
            "loaded_count": len(test_examples),
            "sample_id": test_examples[0].example_id,
            "sample_question": test_examples[0].question[:90] + "...",
            "sample_gold": test_examples[0].gold_answer,
            "sample_options": test_examples[0].options,
        }
    except Exception as e:
        res["medqa_adapter"] = {"status": "FAIL", "error": str(e)}

    # Test PubMedQA Adapter
    try:
        pubmed_adapter = PubMedQAAdapter(data_path=pubmed_dir)
        test_examples = pubmed_adapter.load_split("test", limit=3)
        res["pubmedqa_adapter"] = {
            "status": "PASS",
            "loaded_count": len(test_examples),
            "sample_pmid": test_examples[0].example_id,
            "sample_question": test_examples[0].question[:90] + "...",
            "sample_gold": test_examples[0].gold_answer,
        }
    except Exception as e:
        res["pubmedqa_adapter"] = {"status": "FAIL", "error": str(e)}

    return res


def main():
    print("\n" + "=" * 80)
    print("MEDRAG LOCAL DATASET DIRECTORY INSPECTION AUDIT")
    print("=" * 80)

    # 1. PubMedQA Inspection
    pqa_report = inspect_pubmedqa(PUBMEDQA_DIR)
    print(f"\n[PUBMEDQA AUDIT] Root: {PUBMEDQA_DIR}")
    print(f"Directory Exists:      {pqa_report['exists']}")
    print(f"PQA-L Dataset Found:   {pqa_report['pqal_found']} -> {pqa_report['pqal_path']}")
    print(f"Test GT Found:         {pqa_report['test_gt_found']} -> {pqa_report['test_gt_path']}")
    print(f"Pre-generated Splits:  {pqa_report['generated_splits_found'] if pqa_report['generated_splits_found'] else 'None in download folder (ori_pqal.json and test_ground_truth.json only)'}")
    print(f"Total Expert PQA-L:    {pqa_report['total_pqal_instances']} instances (Expected: 1,000)")
    print(f"Official Test Count:   {pqa_report['test_gt_count']} instances (Overlap with ori_pqal: {pqa_report['test_overlap']})")
    print(f"Label Distribution:    {pqa_report['label_distribution']}")
    print(f"Schema Integrity:      {'PASS (100% valid)' if pqa_report['schema_valid'] else 'FAIL'}")

    # 2. MedQA Inspection
    med_report = inspect_medqa(DATA_CLEAN_DIR)
    print(f"\n[MEDQA AUDIT] Root: {DATA_CLEAN_DIR}")
    print(f"Directory Exists:      {med_report['exists']}")
    print(f"Contains MedQA:        {med_report['is_medqa']} (Identified questions/ and textbooks/ trees)")
    print(f"Regional Subsets:      {med_report['subsets_found']}")
    print(f"English 4-Option Files:{list(med_report['us_4options_splits'].keys())}")
    for s, info in med_report.get("us_4options_splits", {}).items():
        cnt = med_report.get("us_4options_counts", {}).get(s, 0)
        dist = med_report.get("us_4options_option_distributions", {}).get(s, {})
        print(f"   - {s.upper():<5}: {cnt:>6} items ({info['size']:,} bytes) | Options: {dist} | Path: {info['path']}")
    print(f"Textbooks Located:     {len(med_report['textbooks_found'])} official .txt books ({med_report.get('textbooks_total_mb', 0)} MB)")
    print(f"Textbooks Directory:   {med_report.get('textbooks_dir')}")
    print(f"Schema Integrity:      {'PASS (100% valid 4-option items)' if med_report['schema_valid'] else 'FAIL'}")

    # 3. Live Benchmark Adapter Test
    q_dir = str(DATA_CLEAN_DIR / "data_clean" / "questions" / "US" / "4_options")
    tb_dir = med_report.get("textbooks_dir", "")
    p_data_dir = str(Path(pqa_report.get("pqal_path", "")).parent) if pqa_report.get("pqal_path") else ""

    print("\n" + "-" * 80)
    print("[LIVE BENCHMARK ADAPTER INTEGRATION TEST]")
    print("-" * 80)
    adapter_results = test_adapters_live(q_dir, tb_dir, p_data_dir)
    print(f"MedQAAdapter on original download directory:    {adapter_results['medqa_adapter']['status']}")
    if adapter_results['medqa_adapter']['status'] == "PASS":
        m = adapter_results['medqa_adapter']
        print(f"   -> Sample Question ID: {m['sample_id']} | Gold: {m['sample_gold']}")
        print(f"   -> Question Preview:   {m['sample_question']}")
        print(f"   -> 4 Options:          {m['sample_options']}")

    print(f"\nPubMedQAAdapter on original download directory: {adapter_results['pubmedqa_adapter']['status']}")
    if adapter_results['pubmedqa_adapter']['status'] == "PASS":
        p = adapter_results['pubmedqa_adapter']
        print(f"   -> Sample PMID:        {p['sample_pmid']} | Gold: {p['sample_gold']}")
        print(f"   -> Question Preview:   {p['sample_question']}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
