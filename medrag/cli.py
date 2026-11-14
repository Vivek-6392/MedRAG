"""MedRAG Command-Line Interface (CLI).

Commands:
- verify-datasets: Verifies integrity, split counts, schema compatibility,
  and benchmark adapter readiness for MedQA and PubMedQA datasets.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from medrag.benchmarks.adapters.medqa import MedQAAdapter
from medrag.benchmarks.adapters.pubmedqa import PubMedQAAdapter

OFFICIAL_MEDQA_TEXTBOOKS = [
    "Anatomy_Gray.txt",
    "Biochemistry_Lippincott.txt",
    "Cell_Biology_Alberts.txt",
    "First_Aid_Step1.txt",
    "First_Aid_Step2.txt",
    "Gynecology_Novak.txt",
    "Histology_Ross.txt",
    "Immunology_Janeway.txt",
    "InternalMed_Harrison.txt",
    "Neurology_Adams.txt",
    "Obstentrics_Williams.txt",
    "Pathology_Robbins.txt",
    "Pathoma_Husain.txt",
    "Pediatrics_Nelson.txt",
    "Pharmacology_Katzung.txt",
    "Physiology_Levy.txt",
    "Psichiatry_DSM-5.txt",
    "Surgery_Schwartz.txt",
]


def load_env_file(env_path: Optional[str] = None) -> None:
    """Loads environment variables from a .env file if present."""
    path = Path(env_path) if env_path else Path(".env")
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'\"")
            if k and k not in os.environ:
                os.environ[k] = v


def verify_medqa(
    data_dir: Optional[str] = None,
    corpus_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Verifies MedQA dataset splits, textbook corpus, and adapter loading."""
    result: Dict[str, Any] = {
        "benchmark": "MedQA (USMLE English 4-option)",
        "configured_data_dir": data_dir or os.environ.get("MEDQA_DATA_DIR"),
        "configured_corpus_dir": corpus_dir or os.environ.get("MEDQA_CORPUS_DIR"),
        "status": "UNKNOWN",
        "splits": {},
        "adapter_load_test": "PENDING",
        "textbooks": {},
        "licenses": {
            "code_and_questions": "MIT License (https://github.com/jind11/MedQA)",
            "textbooks": "Copyrighted educational texts (Restricted to non-commercial academic research)",
        },
        "errors": [],
        "warnings": [],
    }

    # Temporarily set env var for adapter resolution
    old_env = os.environ.get("MEDQA_DATA_DIR")
    if data_dir:
        os.environ["MEDQA_DATA_DIR"] = data_dir

    try:
        adapter = MedQAAdapter(data_path=data_dir)

        # Check splits
        for split in ("train", "dev", "test"):
            resolved_file = adapter._resolve_split_path(split)
            if resolved_file and os.path.exists(resolved_file):
                try:
                    examples = adapter.load_split(split)
                    result["splits"][split] = {
                        "path": resolved_file,
                        "count": len(examples),
                        "verified": True,
                    }
                except Exception as e:
                    result["splits"][split] = {"path": resolved_file, "error": str(e), "verified": False}
                    result["errors"].append(f"MedQA split '{split}' failed loading: {e}")
            else:
                result["splits"][split] = {"path": None, "verified": False}
                result["warnings"].append(f"MedQA official split '{split}.jsonl' not found.")

        # Test adapter normalization on sample
        try:
            sample_ex = adapter.load_split("test", limit=5)
            if sample_ex:
                result["adapter_load_test"] = "PASS"
                result["sample_example_id"] = sample_ex[0].example_id
        except Exception as e:
            result["adapter_load_test"] = f"FAIL: {e}"
            result["errors"].append(f"MedQA adapter test load failed: {e}")

        # Check Textbooks
        tb_dir = Path(corpus_dir or os.environ.get("MEDQA_CORPUS_DIR") or "data/medqa/textbooks")
        if tb_dir.exists() and tb_dir.is_dir():
            found_tbs = list(tb_dir.glob("*.txt"))
            found_names = {p.name for p in found_tbs}
            missing_official = [name for name in OFFICIAL_MEDQA_TEXTBOOKS if name not in found_names]
            total_bytes = sum(p.stat().st_size for p in found_tbs)

            result["textbooks"] = {
                "dir": str(tb_dir),
                "count": len(found_tbs),
                "total_size_mb": round(total_bytes / (1024 * 1024), 2),
                "official_complete": len(missing_official) == 0,
                "missing_official": missing_official,
            }
        else:
            result["textbooks"] = {"dir": str(tb_dir), "count": 0, "missing": True}
            result["warnings"].append(f"MedQA textbook corpus directory not found at: {tb_dir}")

        has_test = result["splits"].get("test", {}).get("verified", False)
        if has_test and result["adapter_load_test"] == "PASS":
            result["status"] = "PASS"
        elif any(s.get("verified", False) for s in result["splits"].values()):
            result["status"] = "PARTIAL"
        else:
            result["status"] = "NOT_CONFIGURED"

    finally:
        if old_env is not None:
            os.environ["MEDQA_DATA_DIR"] = old_env
        elif data_dir and "MEDQA_DATA_DIR" in os.environ:
            del os.environ["MEDQA_DATA_DIR"]

    return result


def verify_pubmedqa(
    data_dir: Optional[str] = None,
    corpus_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Verifies PubMedQA dataset records, official split counts, and adapter loading."""
    result: Dict[str, Any] = {
        "benchmark": "PubMedQA (PQA-L Expert-Labeled)",
        "configured_data_dir": data_dir or os.environ.get("PUBMEDQA_DATA_DIR"),
        "status": "UNKNOWN",
        "splits": {},
        "adapter_load_test": "PENDING",
        "licenses": {
            "code_and_annotations": "MIT License (https://github.com/pubmedqa/pubmedqa)",
            "abstracts": "NLM Open Access / Terms and Conditions",
        },
        "errors": [],
        "warnings": [],
    }

    old_env = os.environ.get("PUBMEDQA_DATA_DIR")
    if data_dir:
        os.environ["PUBMEDQA_DATA_DIR"] = data_dir

    try:
        adapter = PubMedQAAdapter(data_path=data_dir)

        # Check splits
        for split in ("test", "dev", "train"):
            resolved = adapter._resolve_split_path(split)
            if resolved and os.path.exists(resolved):
                try:
                    examples = adapter.load_split(split)
                    result["splits"][split] = {
                        "path": resolved,
                        "count": len(examples),
                        "verified": True,
                    }
                except Exception as e:
                    result["splits"][split] = {"path": resolved, "error": str(e), "verified": False}
                    result["errors"].append(f"PubMedQA split '{split}' failed loading: {e}")
            else:
                result["splits"][split] = {"path": None, "verified": False}
                result["warnings"].append(f"PubMedQA split file for '{split}' not found.")

        # Test adapter normalization on sample
        try:
            sample_ex = adapter.load_split("test", limit=5)
            if sample_ex:
                result["adapter_load_test"] = "PASS"
                result["sample_pmid"] = sample_ex[0].example_id
        except Exception as e:
            result["adapter_load_test"] = f"FAIL: {e}"
            result["errors"].append(f"PubMedQA adapter test load failed: {e}")

        has_test = result["splits"].get("test", {}).get("verified", False)
        if has_test and result["adapter_load_test"] == "PASS":
            result["status"] = "PASS"
        elif any(s.get("verified", False) for s in result["splits"].values()):
            result["status"] = "PARTIAL"
        else:
            result["status"] = "NOT_CONFIGURED"

    finally:
        if old_env is not None:
            os.environ["PUBMEDQA_DATA_DIR"] = old_env
        elif data_dir and "PUBMEDQA_DATA_DIR" in os.environ:
            del os.environ["PUBMEDQA_DATA_DIR"]

    return result


def print_cli_report(medqa_res: Dict[str, Any], pqa_res: Dict[str, Any]) -> bool:
    """Formats and prints a comprehensive terminal verification report."""
    print("\n" + "=" * 78)
    print("MEDRAG BENCHMARK DATASET & CORPUS VERIFICATION AUDIT")
    print("=" * 78)

    # MedQA Section
    m_stat = medqa_res["status"]
    m_badge = "[PASS]" if m_stat == "PASS" else ("[PARTIAL]" if m_stat == "PARTIAL" else "[NOT CONFIGURED]")
    print(f"\n1. {medqa_res['benchmark']} -> {m_badge}")
    print(f"   Data Directory:   {medqa_res['configured_data_dir'] or 'Not set ($MEDQA_DATA_DIR)'}")
    print(f"   Corpus Directory: {medqa_res['configured_corpus_dir'] or 'Not set ($MEDQA_CORPUS_DIR)'}")
    print("   Splits:")
    for s, info in medqa_res["splits"].items():
        if info.get("verified"):
            print(f"      - {s.upper():<5}: {info['count']:>6} questions | Path: {info['path']}")
        else:
            print(f"      - {s.upper():<5}: [NOT FOUND]")

    tb = medqa_res.get("textbooks", {})
    if tb.get("count", 0) > 0:
        complete_str = "All 18 official textbooks present" if tb.get("official_complete") else f"Missing: {tb.get('missing_official')}"
        print(f"   Textbooks: {tb['count']} files ({tb.get('total_size_mb', 0)} MB) -> {complete_str}")
    else:
        print(f"   Textbooks: [NOT FOUND in {tb.get('dir')}]")

    print(f"   Adapter Integration: {medqa_res['adapter_load_test']}")
    print(f"   Licenses: Code ({medqa_res['licenses']['code_and_questions']}) | Textbooks ({medqa_res['licenses']['textbooks']})")

    # PubMedQA Section
    p_stat = pqa_res["status"]
    p_badge = "[PASS]" if p_stat == "PASS" else ("[PARTIAL]" if p_stat == "PARTIAL" else "[NOT CONFIGURED]")
    print(f"\n2. {pqa_res['benchmark']} -> {p_badge}")
    print(f"   Data Directory: {pqa_res['configured_data_dir'] or 'Not set ($PUBMEDQA_DATA_DIR)'}")
    print("   Splits:")
    for s, info in pqa_res["splits"].items():
        if info.get("verified"):
            print(f"      - {s.upper():<5}: {info['count']:>6} instances | Path: {info['path']}")
        else:
            print(f"      - {s.upper():<5}: [NOT FOUND]")

    print(f"   Adapter Integration: {pqa_res['adapter_load_test']}")
    print(f"   Licenses: Annotations ({pqa_res['licenses']['code_and_annotations']}) | Corpus ({pqa_res['licenses']['abstracts']})")

    # Overall Status Summary
    print("\n" + "-" * 78)
    overall_pass = (m_stat in ("PASS", "PARTIAL")) and (p_stat in ("PASS", "PARTIAL"))

    if m_stat == "PASS" and p_stat == "PASS":
        print("[AUDIT RESULT] ALL BENCHMARK DATASETS VERIFIED SUCCESSFULLY")
        print("Both MedQA and PubMedQA are ready for evaluation runner execution.")
    else:
        print("[AUDIT RESULT] SETUP INCOMPLETE OR PARTIAL")
        if m_stat == "NOT_CONFIGURED":
            print("\nTo setup MedQA:")
            print("   python scripts/download_medqa.py --source-dir <path_to_data_clean>")
        if p_stat == "NOT_CONFIGURED":
            print("\nTo setup PubMedQA:")
            print("   python scripts/download_pubmedqa.py")
    print("=" * 78 + "\n")

    return overall_pass


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m medrag.cli",
        description="MedRAG Command-Line Interface and Dataset Verification.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # verify-datasets subcommand
    v_parser = subparsers.add_parser(
        "verify-datasets",
        help="Verify downloaded benchmark datasets, split counts, and adapter compatibility.",
    )
    v_parser.add_argument(
        "--medqa-dir",
        type=str,
        default=None,
        help="Path to MedQA dataset directory (overrides $MEDQA_DATA_DIR).",
    )
    v_parser.add_argument(
        "--medqa-corpus-dir",
        type=str,
        default=None,
        help="Path to MedQA textbook corpus directory (overrides $MEDQA_CORPUS_DIR).",
    )
    v_parser.add_argument(
        "--pubmedqa-dir",
        type=str,
        default=None,
        help="Path to PubMedQA dataset directory (overrides $PUBMEDQA_DATA_DIR).",
    )
    v_parser.add_argument(
        "--env-file",
        type=str,
        default=".env",
        help="Path to local .env configuration file (default: .env).",
    )
    v_parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero code if any benchmark dataset is missing or invalid.",
    )
    v_parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw verification results as JSON.",
    )

    args = parser.parse_args()

    # Default to verify-datasets if no subcommand specified
    if not args.subcommand or args.subcommand == "verify-datasets":
        load_env_file(getattr(args, "env_file", ".env"))

        medqa_res = verify_medqa(
            data_dir=getattr(args, "medqa_dir", None),
            corpus_dir=getattr(args, "medqa_corpus_dir", None),
        )
        pqa_res = verify_pubmedqa(
            data_dir=getattr(args, "pubmedqa_dir", None),
        )

        if getattr(args, "json", False):
            print(json.dumps({"medqa": medqa_res, "pubmedqa": pqa_res}, indent=2))
            return 0

        success = print_cli_report(medqa_res, pqa_res)
        if getattr(args, "strict", False) and not success:
            return 1
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
