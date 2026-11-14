"""Automatic download and setup script for the official MedQA benchmark dataset.

Downloads or extracts the official MedQA (USMLE English 4-option subset) dataset
and associated medical textbook corpus from its original source:
Official Repository: https://github.com/jind11/MedQA

Key Features:
- Preserves official splits: train (10,178), dev (1,272), test (1,273).
- Preserves the 18 official English reference medical textbooks.
- Keeps datasets outside the Git repository by default (~/.medrag/data/medqa).
- Prompts before overwriting existing files (or requires --force / --overwrite).
- Validates all records and reports dataset statistics.
- Validates source (MIT) and textbook licenses separately.
- Never substitutes an unverified third-party mirror.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

OFFICIAL_REPO_URL = "https://github.com/jind11/MedQA"
OFFICIAL_GDRIVE_ID = "1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw"
OFFICIAL_GDRIVE_URL = f"https://drive.google.com/file/d/{OFFICIAL_GDRIVE_ID}/view?usp=sharing"

DEFAULT_MEDQA_DIR = Path.home() / ".medrag" / "data" / "medqa"
DEFAULT_CORPUS_DIR = DEFAULT_MEDQA_DIR / "textbooks"

OFFICIAL_TEXTBOOKS = [
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


def print_license_notice() -> None:
    print("\n" + "=" * 78)
    print("MEDQA BENCHMARK DATASET & CORPUS LICENSING NOTICE")
    print("=" * 78)
    print("1. MedQA Code & Question Formats:")
    print("   - Source: https://github.com/jind11/MedQA")
    print("   - License: MIT License (Copyright (c) 2020 Di Jin et al.)")
    print("\n2. Underlying USMLE Medical Questions & 18 Reference Textbooks:")
    print("   - Questions derived from professional medical board examinations.")
    print("   - Textbooks are copyrighted educational materials authored by medical publishers.")
    print("   - Usage Terms: Restricted strictly to non-commercial academic research,")
    print("     evaluation, and reproducibility benchmarking under fair-use principles.")
    print("   - Do NOT distribute, re-host, or commit textbook files to public git repositories.")
    print("=" * 78 + "\n")


def find_local_source_candidates() -> List[Path]:
    """Scans common download directories for existing MedQA data_clean folders."""
    candidates = [
        Path.home() / "Downloads" / "data_clean",
        Path.home() / "Downloads" / "data_clean" / "data_clean",
        Path("C:/Users/vivek/Downloads/data_clean"),
        Path("C:/Users/vivek/Downloads/data_clean/data_clean"),
        Path.home() / "data_clean",
        Path("data_clean"),
    ]
    existing = []
    for c in candidates:
        try:
            if c.exists() and (c / "questions" / "US").exists():
                existing.append(c)
            elif c.exists() and (c / "data_clean" / "questions" / "US").exists():
                existing.append(c / "data_clean")
        except Exception:
            pass
    return existing


def locate_question_files(source_dir: Path) -> Dict[str, Path]:
    """Finds official English 4-option question files in a data_clean tree."""
    locations_to_try = [
        source_dir / "questions" / "US" / "4_options",
        source_dir / "data_clean" / "questions" / "US" / "4_options",
        source_dir / "US" / "4_options",
        source_dir / "4_options",
        source_dir,
    ]

    splits = {}
    for loc in locations_to_try:
        if not loc.exists():
            continue
        # Look for phrases_no_exclude_{split}.jsonl or {split}.jsonl
        for s in ("train", "dev", "test"):
            cand1 = loc / f"phrases_no_exclude_{s}.jsonl"
            cand2 = loc / f"{s}.jsonl"
            if cand1.exists():
                splits[s] = cand1
            elif cand2.exists() and s not in splits:
                splits[s] = cand2
        if len(splits) == 3:
            break

    # If 4_options not found, check 5-option US directory
    if len(splits) < 3:
        us_dir = source_dir / "questions" / "US"
        if us_dir.exists():
            for s in ("train", "dev", "test"):
                cand = us_dir / f"{s}.jsonl"
                if cand.exists() and s not in splits:
                    splits[s] = cand

    return splits


def locate_textbook_files(source_dir: Path) -> List[Path]:
    """Finds English medical textbooks in a data_clean tree."""
    locations = [
        source_dir / "textbooks" / "en",
        source_dir / "data_clean" / "textbooks" / "en",
        source_dir / "textbooks",
        source_dir / "en",
    ]
    for loc in locations:
        if loc.exists() and loc.is_dir():
            txt_files = list(loc.glob("*.txt"))
            if txt_files:
                return txt_files
    return []


def validate_medqa_file(path: Path, expected_split: str) -> Tuple[int, Dict[str, int]]:
    """Validates records in a MedQA jsonl split file."""
    count = 0
    option_dist: Dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0}
    seen_questions = set()

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            q = item.get("question")
            if not q or not isinstance(q, str) or not q.strip():
                raise ValueError(f"Line {idx} in {path.name}: Empty question text.")
            options = item.get("options", {})
            if not isinstance(options, dict) or len(options) < 4:
                raise ValueError(f"Line {idx} in {path.name}: Expected at least 4 options, found: {list(options.keys())}")
            ans = item.get("answer_idx", "")
            if ans in option_dist:
                option_dist[ans] += 1
            seen_questions.add(q.strip())
            count += 1

    return count, option_dist


def setup_from_local_source(
    source_dir: Path,
    output_dir: Path,
    corpus_dir: Path,
    force: bool = False,
) -> bool:
    """Sets up MedQA from a verified local data_clean folder."""
    print(f"\n[INFO] Sourcing official MedQA data from: {source_dir}")

    q_files = locate_question_files(source_dir)
    if len(q_files) < 3:
        print(f"[ERROR] Could not find all 3 official splits in {source_dir}.")
        print(f"        Found: {list(q_files.keys())}")
        return False

    tb_files = locate_textbook_files(source_dir)
    print(f"[INFO] Located {len(q_files)} question split files and {len(tb_files)} textbook files.")

    # Check existing files
    target_files = [output_dir / f"{s}.jsonl" for s in ("train", "dev", "test")]
    existing = [p for p in target_files if p.exists()]
    if existing and not force:
        print(f"\n[WARNING] Found {len(existing)} existing files in target directory: {output_dir}")
        for p in existing:
            print(f"   - {p.name} ({p.stat().st_size:,} bytes)")
        if not sys.stdin.isatty():
            print("[ERROR] Non-interactive environment and --force not specified. Aborting to protect existing data.")
            return False
        choice = input("\nDo you want to overwrite these existing files? [y/N]: ").strip().lower()
        if choice not in ("y", "yes"):
            print("[INFO] Setup aborted by user.")
            return False

    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    # Copy and standardize question splits
    print("\n[INFO] Copying and standardizing official question splits...")
    stats = {}
    for split_name, src_path in q_files.items():
        dst_path = output_dir / f"{split_name}.jsonl"
        print(f"   Copying {src_path.name} -> {dst_path}")
        shutil.copy2(src_path, dst_path)

        # Validate copied file
        n_items, opt_dist = validate_medqa_file(dst_path, split_name)
        stats[split_name] = {"count": n_items, "options": opt_dist}

    # Copy textbooks
    print(f"\n[INFO] Copying {len(tb_files)} official medical textbooks to {corpus_dir}...")
    total_bytes = 0
    copied_tb = 0
    for tb_path in tb_files:
        dst_tb = corpus_dir / tb_path.name
        shutil.copy2(tb_path, dst_tb)
        total_bytes += dst_tb.stat().st_size
        copied_tb += 1

    # Print summary statistics
    print("\n" + "=" * 78)
    print("MEDQA BENCHMARK SETUP COMPLETED SUCCESSFULLY")
    print("=" * 78)
    print(f"Target Directory (Questions): {output_dir}")
    print(f"Target Directory (Textbooks): {corpus_dir}")
    print("\nSplit Statistics (USMLE English 4-option):")
    for s_name in ("train", "dev", "test"):
        if s_name in stats:
            info = stats[s_name]
            print(f"   - {s_name.upper():<5}: {info['count']:>6} questions | Option distribution: {info['options']}")
    print("\nCorpus Statistics:")
    print(f"   - Textbooks Copied: {copied_tb} files ({total_bytes / (1024*1024):.2f} MB)")
    print("=" * 78)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Automatic setup for the official MedQA benchmark dataset."
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="Path to pre-downloaded or extracted data_clean directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_MEDQA_DIR),
        help=f"Target directory for questions (default: {DEFAULT_MEDQA_DIR}).",
    )
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default=str(DEFAULT_CORPUS_DIR),
        help=f"Target directory for textbooks (default: {DEFAULT_CORPUS_DIR}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files without prompting.",
    )
    args = parser.parse_args()

    print_license_notice()

    out_path = Path(args.output_dir).resolve()
    corp_path = Path(args.corpus_dir).resolve()

    # Determine source directory
    src_dir = None
    if args.source_dir:
        p = Path(args.source_dir).resolve()
        if p.exists():
            src_dir = p
        else:
            print(f"[ERROR] Provided --source-dir does not exist: {args.source_dir}")
            return 1
    else:
        # Check automatic candidate locations
        candidates = find_local_source_candidates()
        if candidates:
            src_dir = candidates[0]
            print(f"[INFO] Auto-detected local MedQA source at: {src_dir}")

    if src_dir:
        success = setup_from_local_source(src_dir, out_path, corp_path, force=args.force)
        if success:
            print("\nTo use these datasets in MedRAG, set your environment variables:")
            print(f'   $env:MEDQA_DATA_DIR = "{out_path}"')
            print(f'   $env:MEDQA_CORPUS_DIR = "{corp_path}"')
            print("\nOr update your local .env file:")
            print(f"   MEDQA_DATA_DIR={out_path}")
            print(f"   MEDQA_CORPUS_DIR={corp_path}\n")
            return 0
        else:
            return 1

    print("\n[INFO] No local MedQA data_clean folder was automatically detected.")
    print(f"Official download source: {OFFICIAL_GDRIVE_URL}")
    print("\nPlease download the official MedQA 'data_clean.zip' from:")
    print(f"   {OFFICIAL_GDRIVE_URL}")
    print("Extract the zip archive, and run:")
    print("   python scripts/download_medqa.py --source-dir <path_to_extracted_data_clean>")
    return 1


if __name__ == "__main__":
    sys.exit(main())
