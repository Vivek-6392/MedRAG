"""Tests for dataset setup scripts and CLI verification command."""

from __future__ import annotations

import json
from pathlib import Path

from medrag.cli import OFFICIAL_MEDQA_TEXTBOOKS, verify_medqa
from scripts.download_medqa import (
    locate_question_files,
    locate_textbook_files,
    setup_from_local_source,
    validate_medqa_file,
)
from scripts.download_pubmedqa import (
    create_split_files,
    validate_pubmedqa_records,
)


def test_medqa_locate_question_files(tmp_path: Path):
    us_dir = tmp_path / "questions" / "US" / "4_options"
    us_dir.mkdir(parents=True)
    (us_dir / "phrases_no_exclude_train.jsonl").write_text("{}", encoding="utf-8")
    (us_dir / "phrases_no_exclude_dev.jsonl").write_text("{}", encoding="utf-8")
    (us_dir / "phrases_no_exclude_test.jsonl").write_text("{}", encoding="utf-8")

    splits = locate_question_files(tmp_path)
    assert len(splits) == 3
    assert "train" in splits
    assert "dev" in splits
    assert "test" in splits


def test_medqa_locate_textbook_files(tmp_path: Path):
    tb_dir = tmp_path / "textbooks" / "en"
    tb_dir.mkdir(parents=True)
    for name in OFFICIAL_MEDQA_TEXTBOOKS[:3]:
        (tb_dir / name).write_text("Medical knowledge text...", encoding="utf-8")

    tbs = locate_textbook_files(tmp_path)
    assert len(tbs) == 3


def test_medqa_validation_and_setup(tmp_path: Path):
    source_dir = tmp_path / "source"
    us_dir = source_dir / "questions" / "US" / "4_options"
    us_dir.mkdir(parents=True)
    tb_dir = source_dir / "textbooks" / "en"
    tb_dir.mkdir(parents=True)

    # Write sample 4-option items
    sample_item = {
        "question": "What is the primary intervention for acute heat stroke?",
        "options": {"A": "Cold water immersion", "B": "Acetaminophen", "C": "Dantrolene", "D": "Saline"},
        "answer_idx": "A",
    }
    for s in ("train", "dev", "test"):
        with open(us_dir / f"phrases_no_exclude_{s}.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps(sample_item) + "\n")

    for name in OFFICIAL_MEDQA_TEXTBOOKS[:2]:
        (tb_dir / name).write_text("Medical reference...", encoding="utf-8")

    out_dir = tmp_path / "target_medqa"
    corpus_dir = out_dir / "textbooks"

    # Setup with force=True
    success = setup_from_local_source(source_dir, out_dir, corpus_dir, force=True)
    assert success is True
    assert (out_dir / "train.jsonl").exists()
    assert (out_dir / "test.jsonl").exists()
    assert (corpus_dir / OFFICIAL_MEDQA_TEXTBOOKS[0]).exists()

    # Validate output file
    count, opt_dist = validate_medqa_file(out_dir / "test.jsonl", "test")
    assert count == 1
    assert opt_dist["A"] == 1


def test_pubmedqa_validation_and_splitting(tmp_path: Path):
    data_dir = tmp_path / "pubmedqa_test"
    data_dir.mkdir(parents=True)

    # Sample ori_pqal
    sample_pqal = {
        "1001": {
            "QUESTION": "Does drug X improve cardiac outcomes?",
            "CONTEXTS": ["In a trial of 200 patients, drug X reduced mortality by 30%."],
            "final_decision": "yes",
        },
        "1002": {
            "QUESTION": "Is routine biopsy indicated for asymptomatic simple cysts?",
            "CONTEXTS": ["Ultrasound features reliably differentiate simple from complex renal cysts."],
            "final_decision": "no",
        },
        "1003": {
            "QUESTION": "Can biomarker Y predict chemotherapy resistance?",
            "CONTEXTS": ["Current evidence shows inconclusive correlation with resistance."],
            "final_decision": "maybe",
        },
    }
    pqal_file = data_dir / "ori_pqal.json"
    with open(pqal_file, "w", encoding="utf-8") as f:
        json.dump(sample_pqal, f)

    # Sample test ground truth (pmid 1001 is test)
    gt_file = data_dir / "test_ground_truth.json"
    with open(gt_file, "w", encoding="utf-8") as f:
        json.dump({"1001": "yes"}, f)

    total, label_dist, test_cnt = validate_pubmedqa_records(pqal_file, gt_file)
    assert total == 3
    assert test_cnt == 1
    assert label_dist["yes"] == 1
    assert label_dist["no"] == 1
    assert label_dist["maybe"] == 1

    # Split generation
    create_split_files(pqal_file, gt_file, data_dir)
    assert (data_dir / "test.json").exists()
    assert (data_dir / "dev.json").exists()
    assert (data_dir / "train.json").exists()

    with open(data_dir / "test.json", "r", encoding="utf-8") as f:
        test_items = json.load(f)
    assert len(test_items) == 1
    assert "1001" in test_items


def test_cli_verification_logic(tmp_path: Path):
    medqa_dir = tmp_path / "medqa"
    medqa_dir.mkdir(parents=True)
    tb_dir = medqa_dir / "textbooks"
    tb_dir.mkdir(parents=True)

    # Empty dirs should report NOT_CONFIGURED or missing
    res_medqa = verify_medqa(data_dir=str(medqa_dir), corpus_dir=str(tb_dir))
    assert res_medqa["status"] in ("NOT_CONFIGURED", "PARTIAL")

    # Add valid split
    sample_item = {
        "question": "What is the primary intervention for acute heat stroke?",
        "options": {"A": "Cooling", "B": "Acetaminophen", "C": "Dantrolene", "D": "Saline"},
        "answer_idx": "A",
    }
    with open(medqa_dir / "test.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(sample_item) + "\n")

    res_medqa_updated = verify_medqa(data_dir=str(medqa_dir), corpus_dir=str(tb_dir))
    assert res_medqa_updated["adapter_load_test"] == "PASS"
    assert res_medqa_updated["splits"]["test"]["verified"] is True
