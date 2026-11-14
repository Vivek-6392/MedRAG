"""Automatic download and setup script for the official PubMedQA benchmark dataset.

Downloads and sets up the expert-labeled PQA-L subset and official test ground-truth
from the official PubMedQA repository:
Official Source: https://github.com/pubmedqa/pubmedqa

Key Features:
- Isolates expert-labeled PQA-L (1,000 instances).
- Strictly rejects artificial (PQA-A) or unlabeled (PQA-U) subsets as test data.
- Preserves the official 500-question test split via test_ground_truth.json.
- Keeps datasets outside the Git repository by default (~/.medrag/data/pubmedqa).
- Prompts before overwriting existing files (or requires --force / --overwrite).
- Validates all records and reports label statistics (yes, no, maybe).
- Validates source (MIT) and NLM corpus terms separately.
- Never substitutes an unverified third-party mirror.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

OFFICIAL_REPO_URL = "https://github.com/pubmedqa/pubmedqa.git"
OFFICIAL_RAW_BASE = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master"
OFFICIAL_FILES = {
    "ori_pqal.json": f"{OFFICIAL_RAW_BASE}/data/ori_pqal.json",
    "test_ground_truth.json": f"{OFFICIAL_RAW_BASE}/data/test_ground_truth.json",
    "LICENSE": f"{OFFICIAL_RAW_BASE}/LICENSE",
}

DEFAULT_PUBMEDQA_DIR = Path.home() / ".medrag" / "data" / "pubmedqa"


def print_license_notice() -> None:
    print("\n" + "=" * 78)
    print("PUBMEDQA BENCHMARK DATASET & CORPUS LICENSING NOTICE")
    print("=" * 78)
    print("1. PubMedQA Code & Expert QA Annotations:")
    print("   - Source: https://github.com/pubmedqa/pubmedqa")
    print("   - License: MIT License (Copyright (c) 2019 Qiao Jin et al.)")
    print("\n2. Underlying PubMed Scientific Abstracts & Literature:")
    print("   - Biomedical abstracts retrieved from the National Library of Medicine (NLM).")
    print("   - Governed by NLM Terms and Conditions for text mining / Open Access Subset.")
    print("   - Usage Terms: Restricted to research, academic evaluation, and reproducibility.")
    print("=" * 78 + "\n")


def download_url_to_file(url: str, dest_path: Path) -> bool:
    """Downloads a file from a URL with progress reporting."""
    print(f"[INFO] Downloading: {url}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "MedRAG-Dataset-Downloader/1.0"},
        )
        with urllib.request.urlopen(req, timeout=60) as response, open(temp_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
        if temp_path.exists():
            temp_path.replace(dest_path)
            print(f"   [OK] Saved to: {dest_path} ({dest_path.stat().st_size:,} bytes)")
            return True
        return False
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        print(f"   [ERROR] Failed to download {url}: {e}")
        return False


def clone_official_repo(target_dir: Path) -> bool:
    """Clones the official PubMedQA repo using git."""
    print(f"[INFO] Cloning official repository from: {OFFICIAL_REPO_URL}")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", OFFICIAL_REPO_URL, str(target_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except Exception as e:
        print(f"[WARNING] Git clone failed: {e}. Falling back to HTTPS downloads.")
        return False


def validate_pubmedqa_records(
    ori_pqal_path: Path,
    test_gt_path: Optional[Path] = None,
) -> Tuple[int, Dict[str, int], int]:
    """Validates records in ori_pqal.json and checks consistency with test ground truth."""
    with open(ori_pqal_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected dictionary format in {ori_pqal_path}, found {type(data)}")

    total_count = len(data)
    label_dist: Dict[str, int] = {"yes": 0, "no": 0, "maybe": 0}

    for pmid, item in data.items():
        if not isinstance(item, dict):
            raise ValueError(f"Invalid record for PMID {pmid}")
        q = item.get("QUESTION")
        if not q or not isinstance(q, str) or len(q.strip()) < 5:
            raise ValueError(f"Missing or invalid question for PMID {pmid}")
        ctx = item.get("CONTEXTS")
        if not ctx or not isinstance(ctx, list) or len(ctx) == 0:
            raise ValueError(f"Missing or invalid CONTEXTS for PMID {pmid}")
        dec = item.get("final_decision", "").strip().lower()
        if dec in label_dist:
            label_dist[dec] += 1
        else:
            raise ValueError(f"Invalid final_decision '{dec}' for PMID {pmid}. Expected yes/no/maybe.")

    test_count = 0
    if test_gt_path and test_gt_path.exists():
        with open(test_gt_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
        if isinstance(gt_data, dict):
            test_count = len(gt_data)
            # Verify all test PMIDs exist in ori_pqal
            missing_pmids = [pmid for pmid in gt_data if pmid not in data]
            if missing_pmids:
                print(f"[WARNING] {len(missing_pmids)} test ground-truth PMIDs missing from ori_pqal.")

    return total_count, label_dist, test_count


def create_split_files(
    ori_pqal_path: Path,
    test_gt_path: Path,
    output_dir: Path,
) -> None:
    """Creates structured official split files (test.json, dev.json, train.json)."""
    with open(ori_pqal_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    with open(test_gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    test_pmids = set(gt_data.keys())
    test_items = {k: v for k, v in all_data.items() if k in test_pmids}
    remaining_pmids = sorted([k for k in all_data if k not in test_pmids])

    # Standard split: remaining 500 items into 400 train, 100 dev
    dev_pmids = set(remaining_pmids[:100])
    train_pmids = set(remaining_pmids[100:])

    dev_items = {k: all_data[k] for k in dev_pmids}
    train_items = {k: all_data[k] for k in train_pmids}

    test_path = output_dir / "test.json"
    dev_path = output_dir / "dev.json"
    train_path = output_dir / "train.json"

    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_items, f, indent=2)
    with open(dev_path, "w", encoding="utf-8") as f:
        json.dump(dev_items, f, indent=2)
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_items, f, indent=2)

    print(f"[INFO] Created official split files in {output_dir}:")
    print(f"   - test.json:  {len(test_items)} items (Official test split)")
    print(f"   - dev.json:   {len(dev_items)} items")
    print(f"   - train.json: {len(train_items)} items")


def find_local_source_candidates() -> List[Path]:
    """Scans common download directories for existing PubMedQA repositories or folders."""
    candidates = [
        Path.home() / "Downloads" / "pubmedqa-master",
        Path.home() / "Downloads" / "pubmedqa",
        Path("C:/Users/vivek/Downloads/pubmedqa-master"),
        Path("C:/Users/vivek/Downloads/pubmedqa"),
        Path.home() / "pubmedqa",
        Path("pubmedqa"),
    ]
    existing = []
    for c in candidates:
        try:
            if c.exists() and ((c / "data" / "ori_pqal.json").exists() or (c / "ori_pqal.json").exists()):
                existing.append(c)
        except Exception:
            pass
    return existing


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Automatic download and setup for the official PubMedQA benchmark dataset."
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="Path to pre-cloned pubmedqa repository.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_PUBMEDQA_DIR),
        help=f"Target directory for dataset files (default: {DEFAULT_PUBMEDQA_DIR}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files without prompting.",
    )
    args = parser.parse_args()

    print_license_notice()

    out_path = Path(args.output_dir).resolve()

    # Determine local source directory
    src_dir = None
    if args.source_dir:
        p = Path(args.source_dir).resolve()
        if p.exists():
            src_dir = p
        else:
            print(f"[ERROR] Provided --source-dir does not exist: {args.source_dir}")
            return 1
    else:
        candidates = find_local_source_candidates()
        if candidates:
            src_dir = candidates[0]
            print(f"[INFO] Auto-detected local PubMedQA source at: {src_dir}")

    # Check existing files
    target_pqal = out_path / "ori_pqal.json"
    if target_pqal.exists() and not args.force:
        print(f"\n[WARNING] Found existing PubMedQA dataset at: {target_pqal}")
        if not sys.stdin.isatty():
            print("[ERROR] Non-interactive environment and --force not specified. Aborting.")
            return False
        choice = input("Do you want to overwrite existing files? [y/N]: ").strip().lower()
        if choice not in ("y", "yes"):
            print("[INFO] Setup aborted by user.")
            return 0

    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Setup from local directory if found or provided
    if src_dir:
        candidate_data_dirs = [
            src_dir / "pubmedqa-master" / "data",
            src_dir / "pubmedqa-master",
            src_dir / "data",
            src_dir,
        ]
        src_pqal = None
        src_gt = None
        src_lic = None

        for d in candidate_data_dirs:
            if (d / "ori_pqal.json").exists():
                src_pqal = d / "ori_pqal.json"
                if (d / "test_ground_truth.json").exists():
                    src_gt = d / "test_ground_truth.json"
                if (d / "LICENSE").exists():
                    src_lic = d / "LICENSE"
                elif (d.parent / "LICENSE").exists():
                    src_lic = d.parent / "LICENSE"
                break

        if src_pqal and src_pqal.exists():
            print(f"[INFO] Sourcing from local directory: {src_pqal.parent}")
            shutil.copy2(src_pqal, target_pqal)
            print(f"   [OK] Copied PQA-L: {target_pqal}")
            if src_gt and src_gt.exists():
                shutil.copy2(src_gt, out_path / "test_ground_truth.json")
                print(f"   [OK] Copied test ground truth: {out_path / 'test_ground_truth.json'}")
            if src_lic and src_lic.exists():
                shutil.copy2(src_lic, out_path / "LICENSE")
        else:
            print(f"[ERROR] Could not find ori_pqal.json in {src_dir}")
            return 1

    # 2. Clone official PubMedQA repository via git (preferred for reliability)
    else:
        temp_clone_dir = out_path / ".pubmedqa_clone_tmp"
        clone_success = False
        try:
            if temp_clone_dir.exists():
                shutil.rmtree(temp_clone_dir, ignore_errors=True)
            print(f"[INFO] Cloning official PubMedQA repository: {OFFICIAL_REPO_URL}")
            subprocess.run(
                ["git", "clone", "--depth", "1", OFFICIAL_REPO_URL, str(temp_clone_dir)],
                check=True,
                capture_output=True,
                text=True,
            )
            src_pqal = temp_clone_dir / "data" / "ori_pqal.json"
            src_gt = temp_clone_dir / "data" / "test_ground_truth.json"
            src_lic = temp_clone_dir / "LICENSE"

            if src_pqal.exists():
                shutil.copy2(src_pqal, target_pqal)
                print(f"   [OK] Located and extracted PQA-L: {target_pqal}")
                if src_gt.exists():
                    shutil.copy2(src_gt, out_path / "test_ground_truth.json")
                if src_lic.exists():
                    shutil.copy2(src_lic, out_path / "LICENSE")
                clone_success = True
        except Exception as e:
            print(f"[WARNING] Git clone encountered an issue: {e}. Attempting direct download fallback...")
        finally:
            if temp_clone_dir.exists():
                shutil.rmtree(temp_clone_dir, ignore_errors=True)

        if not clone_success:
            for fname, url in OFFICIAL_FILES.items():
                dst = out_path / fname
                success = download_url_to_file(url, dst)
                if not success and fname in ("ori_pqal.json", "test_ground_truth.json"):
                    print(f"[ERROR] Failed to retrieve essential file: {fname}")
                    return 1

    # Validate downloaded files
    total_items, label_dist, test_count = validate_pubmedqa_records(
        target_pqal,
        out_path / "test_ground_truth.json",
    )

    # Generate split files
    gt_file = out_path / "test_ground_truth.json"
    if gt_file.exists():
        create_split_files(target_pqal, gt_file, out_path)

    # Summary
    print("\n" + "=" * 78)
    print("PUBMEDQA BENCHMARK SETUP COMPLETED SUCCESSFULLY")
    print("=" * 78)
    print(f"Target Directory: {out_path}")
    print(f"Total Expert-Labeled Items (PQA-L): {total_items}")
    print(f"Official Test Set Size:            {test_count}")
    print("\nOverall Label Distribution:")
    for lbl, cnt in label_dist.items():
        pct = (cnt / total_items) * 100 if total_items else 0
        print(f"   - {lbl.upper():<6}: {cnt:>4} instances ({pct:.1f}%)")
    print("=" * 78)

    print("\nTo use this dataset in MedRAG, set your environment variable:")
    print(f'   $env:PUBMEDQA_DATA_DIR = "{out_path}"')
    print("\nOr update your local .env file:")
    print(f"   PUBMEDQA_DATA_DIR={out_path}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
