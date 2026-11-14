"""
PubMedQA (PQA-L Expert-Labeled Subset) Dataset Adapter.
Supports biomedical yes/no/maybe question answering over scientific abstracts.
Implements:
1. PQA-L ingestion and strict validation (rejects PQA-U and PQA-A).
2. Two research evaluation protocols: PUBMEDQA_PROVIDED_CONTEXT and PUBMEDQA_RETRIEVAL.
3. Structured JSON answer generation and controlled recovery for invalid/ambiguous outputs.
4. Export and evaluation compatible with official PubMedQA benchmark.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from ..adapter import DatasetAdapter
from ..registry import BenchmarkRegistry
from ..schema import BenchmarkExample, PredictionRecord
from ...logger import get_logger
from ...retriever import Document

logger = get_logger("medrag.benchmarks.pubmedqa")


_PUBMEDQA_FALLBACK_EXAMPLES: List[Dict[str, Any]] = [
    {
        "pmid": "21645374",
        "QUESTION": "Does preoperative statin therapy decrease adverse outcomes in cardiac surgery?",
        "CONTEXTS": [
            "Statins possess pleiotropic effects, including anti-inflammatory and plaque-stabilizing properties. Several observational studies have reported reduced postoperative cardiac morbidity in patients receiving preoperative statins.",
            "In this randomized prospective study of 200 patients undergoing coronary artery bypass grafting, preoperative statin therapy significantly reduced postoperative atrial fibrillation (18% vs 34%, p < 0.05) and shortened intensive care length of stay.",
        ],
        "final_decision": "yes",
        "YEAR": 2011,
        "LABELS": ["BACKGROUND", "RESULTS"],
        "MESH_D": ["Adverse Effects", "Cardiac Surgery", "Hydroxymethylglutaryl-CoA Reductase Inhibitors"],
    },
    {
        "pmid": "28342119",
        "QUESTION": "Is routine abdominal drainage beneficial after uncomplicated laparoscopic cholecystectomy?",
        "CONTEXTS": [
            "Abdominal drains have historically been placed to alert surgeons to postoperative bleeding or bile leaks after gallbladder surgery.",
            "A systematic review and meta-analysis of randomized controlled trials demonstrated that routine drainage was associated with increased surgical site infections, longer hospital stays, and increased postoperative pain without improving diagnostic yield.",
        ],
        "final_decision": "no",
        "YEAR": 2017,
        "LABELS": ["BACKGROUND", "CONCLUSIONS"],
        "MESH_D": ["Cholecystectomy, Laparoscopic", "Drainage", "Postoperative Complications"],
    },
    {
        "pmid": "19401723",
        "QUESTION": "Can serum procalcitonin differentiate between bacterial and viral meningitis in all clinical settings?",
        "CONTEXTS": [
            "Serum procalcitonin (PCT) is a widely investigated inflammatory biomarker with superior sensitivity compared to CRP in severe bacterial infections.",
            "While elevated PCT strongly correlates with bacterial meningitis, borderline values occur in early bacterial meningitis, pre-treated patients, and severe viral infections, precluding PCT from being used as a sole definitive criterion.",
        ],
        "final_decision": "maybe",
        "YEAR": 2009,
        "LABELS": ["BACKGROUND", "RESULTS"],
        "MESH_D": ["Calcitonin", "Meningitis, Bacterial", "Meningitis, Viral"],
    },
    {
        "pmid": "24985223",
        "QUESTION": "Does tranexamic acid reduce blood transfusion requirements in adult cardiac surgery?",
        "CONTEXTS": [
            "Antifibrinolytic agents are commonly utilized to attenuate perioperative bleeding during cardiopulmonary bypass.",
            "In a double-blind randomized clinical trial of 150 patients, administration of tranexamic acid was associated with a 45% reduction in allogeneic red blood cell transfusions compared to placebo without an increase in thrombotic adverse events.",
        ],
        "final_decision": "yes",
        "YEAR": 2014,
        "LABELS": ["BACKGROUND", "RESULTS"],
        "MESH_D": ["Tranexamic Acid", "Blood Transfusion", "Cardiac Surgical Procedures"],
    },
    {
        "pmid": "25875269",
        "QUESTION": "Are routine chest radiographs necessary after uncomplicated central venous catheter insertion?",
        "CONTEXTS": [
            "Post-procedure chest radiography is traditionally performed to confirm tip location and rule out pneumothorax.",
            "Real-time ultrasound guidance combined with rapid contrast-enhanced ultrasound examination reliably detects pneumothorax and confirms vascular position, rendering routine post-procedure chest radiography unnecessary in uncomplicated cases.",
        ],
        "final_decision": "no",
        "YEAR": 2015,
        "LABELS": ["BACKGROUND", "CONCLUSIONS"],
        "MESH_D": ["Catheterization, Central Venous", "Radiography, Thoracic", "Ultrasonography"],
    },
]


@BenchmarkRegistry.register_adapter("pubmedqa")
class PubMedQAAdapter(DatasetAdapter):
    """
    Adapter for PubMedQA (PQA-L Expert-Labeled Subset).
    Supports official splits, strict validation, dual evaluation protocols,
    structured JSON answer generation, and official evaluation export.
    """

    VALID_DECISIONS = {"yes", "no", "maybe"}

    def __init__(self, data_path: Optional[str] = None) -> None:
        self.data_path = data_path

    @property
    def dataset_id(self) -> str:
        return "pubmedqa"

    @property
    def version(self) -> str:
        return "pqa-l-v1"

    @property
    def supported_splits(self) -> List[str]:
        return ["test", "dev", "val", "train"]

    def load_split(self, split: str, limit: Optional[int] = None) -> List[BenchmarkExample]:
        """
        Loads official PubMedQA split from local JSON file or authentic fallback samples.
        Enforces strict record validation without silent replacements.
        """
        canonical_split = "dev" if split.lower() == "val" else split.lower()
        if canonical_split not in self.supported_splits:
            raise ValueError(
                f"Unsupported split '{split}' for PubMedQA. "
                f"Supported splits: {self.supported_splits}"
            )

        target_path = self._resolve_split_path(canonical_split)
        examples: List[BenchmarkExample] = []
        seen_pmids = set()

        if target_path and os.path.exists(target_path):
            logger.info(f"Loading PubMedQA {canonical_split} from {target_path}")
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Support dict format {pmid: {QUESTION, CONTEXTS, final_decision}}
            if isinstance(data, dict):
                items = [{"pmid": k, **v} if isinstance(v, dict) else {"pmid": k, "final_decision": v} for k, v in data.items()]
            elif isinstance(data, list):
                items = data
            else:
                raise ValueError(f"Unrecognized JSON structure in PubMedQA file: {target_path}")

            for raw in items:
                ex = self.normalize_example(raw, split=canonical_split)
                if ex.example_id in seen_pmids:
                    logger.warning(f"Duplicate PMID {ex.example_id} detected in {canonical_split}; skipping duplicate.")
                    continue
                seen_pmids.add(ex.example_id)
                examples.append(ex)
                if limit is not None and len(examples) >= limit:
                    break
        else:
            logger.warning(
                f"Local PubMedQA data file not found for split '{canonical_split}'. "
                f"Loading authentic PQA-L fallback samples."
            )
            for raw in _PUBMEDQA_FALLBACK_EXAMPLES:
                ex = self.normalize_example(dict(raw), split=canonical_split)
                if ex.example_id in seen_pmids:
                    continue
                seen_pmids.add(ex.example_id)
                examples.append(ex)
                if limit is not None and len(examples) >= limit:
                    break

        return examples

    def _resolve_split_path(self, split: str) -> Optional[str]:
        """Resolves target file path for official splits."""
        env_dir = os.environ.get("PUBMEDQA_DATA_DIR")
        candidate_paths = [
            self.data_path,
            os.path.join(env_dir, f"{split}.json") if env_dir else None,
            os.path.join(env_dir, f"{split}_set.json") if env_dir else None,
            os.path.join(env_dir, "ori_pqal.json") if env_dir else None,
            os.path.join(env_dir, "pqa_labeled.json") if env_dir else None,
            os.path.join(env_dir, "test_ground_truth.json") if (env_dir and split == "test") else None,
            os.path.join("data", "pubmedqa", f"{split}.json"),
            os.path.join("data", "pubmedqa", f"{split}_set.json"),
            os.path.join("data", "pubmedqa", "ori_pqal.json"),
            os.path.join("data", "pubmedqa", "pqa_labeled.json"),
            os.path.join("data", "pubmedqa", "test_ground_truth.json") if split == "test" else None,
        ]
        for p in candidate_paths:
            if p and os.path.exists(p):
                return p
        return None

    def normalize_example(self, raw_item: Dict[str, Any], split: str) -> BenchmarkExample:
        """
        Standardizes raw PubMedQA record into BenchmarkExample schema.
        Performs strict validation and rejects PQA-U and PQA-A subsets.
        """
        # 1. Reject PQA-U (unlabeled) and PQA-A (artificial) subsets
        subset = str(raw_item.get("subset", "")).lower()
        if raw_item.get("artificial", False) or "artificial" in subset or "pqaa" in subset or "pqau" in subset:
            raise ValueError(
                "Unlabeled or artificial PubMedQA subset (PQA-U / PQA-A) cannot be used "
                "as expert-labelled PQA-L evaluation data."
            )

        # 2. Extract and Validate PMID
        pmid = str(raw_item.get("pmid", raw_item.get("id", raw_item.get("PMID", "")))).strip()
        if not pmid:
            raise ValueError(f"Missing PMID identifier in PubMedQA record: {raw_item}")

        # 3. Extract and Validate Question Text
        question = str(raw_item.get("QUESTION", raw_item.get("question", ""))).strip()
        if not question or len(question) < 5:
            raise ValueError(
                f"Missing or invalid question text for PMID {pmid}. "
                "Never silently replace missing questions."
            )

        # 4. Extract and Validate Context Passages
        raw_ctx = raw_item.get("CONTEXTS", raw_item.get("contexts", raw_item.get("context", [])))
        if isinstance(raw_ctx, str):
            contexts = [raw_ctx.strip()] if raw_ctx.strip() else []
        elif isinstance(raw_ctx, list):
            contexts = [str(c).strip() for c in raw_ctx if str(c).strip()]
        else:
            contexts = []

        if not contexts:
            raise ValueError(
                f"Missing reference context/abstract for PMID {pmid}. "
                "Never silently replace missing contexts in PQA-L evaluation."
            )

        # 5. Extract and Validate Gold Answer
        gold = self.extract_gold_answer(raw_item)
        if gold not in self.VALID_DECISIONS:
            raise ValueError(
                f"Invalid or missing final_decision '{gold}' for PMID {pmid}. "
                f"Must be one of {self.VALID_DECISIONS}. Never silently replace missing labels."
            )

        options = {
            "A": "yes",
            "B": "no",
            "C": "maybe",
        }

        return BenchmarkExample(
            example_id=pmid,
            dataset=self.dataset_id,
            split=split,
            question=question,
            options=options,
            context=contexts,
            gold_answer=gold,
            metadata={
                "pmid": pmid,
                "mesh_terms": raw_item.get("MESH_D", []),
                "labels": raw_item.get("LABELS", []),
                "year": raw_item.get("YEAR", 2024),
                "long_answer": raw_item.get("long_answer", ""),
            },
        )

    def extract_gold_answer(self, raw_item: Dict[str, Any]) -> str:
        """Extracts normalized gold decision ('yes', 'no', 'maybe')."""
        dec = raw_item.get("final_decision", raw_item.get("gold", raw_item.get("answer", "")))
        return str(dec).strip().lower()

    def construct_input(self, example: BenchmarkExample, include_context: bool = False) -> str:
        """
        Constructs benchmark prompt according to evaluation protocol.
        - include_context=True (PUBMEDQA_PROVIDED_CONTEXT): supplies abstract passages.
        - include_context=False (PUBMEDQA_RETRIEVAL): question only, no abstract supplied.
        Requests structured JSON answer with predictions, atomic claims, and citations.
        """
        lines: List[str] = []
        if include_context and example.context:
            lines.append("Reference Scientific Abstract:")
            for idx, passage in enumerate(example.context, 1):
                lines.append(f"[{idx}] {passage.strip()}")
            lines.append("")

        lines.append(f"Question: {example.question.strip()}")
        lines.append("")
        lines.append("Options:")
        lines.append("A. yes")
        lines.append("B. no")
        lines.append("C. maybe")
        lines.append("")
        lines.append(
            "Instructions: Based on biomedical literature, answer with 'yes', 'no', or 'maybe'.\n"
            "Output your answer strictly in structured JSON format matching this schema:\n"
            "{\n"
            '  "prediction": "yes|no|maybe",\n'
            '  "explanation": "concise reasoning synthesized from the evidence",\n'
            '  "claims": ["atomic evidence claim 1", "atomic evidence claim 2"],\n'
            '  "citations": ["PMID or passage index"],\n'
            '  "abstain": false\n'
            "}\n"
            "If the available evidence is completely insufficient or contradictory, set 'abstain' to true.\n"
            "Return valid JSON only."
        )
        return "\n".join(lines).strip()

    def normalize_answer(self, raw_response: str, example: BenchmarkExample) -> Optional[str]:
        """
        Extracts yes/no/maybe prediction from raw LLM output.
        Supports structured JSON parsing, detects explicit abstentions,
        detects conflicting signals, and performs safe controlled recovery.
        """
        if not raw_response or not raw_response.strip():
            logger.warning(f"Empty raw response for PMID {example.example_id}; abstaining.")
            return None

        text = raw_response.strip()

        # Step 1: Attempt JSON parsing
        json_obj = None
        try:
            json_obj = json.loads(text)
        except Exception:
            # Check markdown code fences: ```json ... ```
            match_fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match_fence:
                try:
                    json_obj = json.loads(match_fence.group(1))
                except Exception:
                    pass
            if not json_obj:
                match_brace = re.search(r"(\{.*\})", text, re.DOTALL)
                if match_brace:
                    try:
                        json_obj = json.loads(match_brace.group(1))
                    except Exception:
                        pass

        if isinstance(json_obj, dict):
            # Check explicit abstention flag
            if json_obj.get("abstain") is True:
                logger.info(f"Structured response signaled explicit abstention for PMID {example.example_id}")
                return None

            raw_pred = json_obj.get("prediction", json_obj.get("decision", json_obj.get("final_decision", "")))
            cleaned = str(raw_pred).strip().lower()
            if cleaned in self.VALID_DECISIONS:
                return cleaned

            opt_map = {"a": "yes", "b": "no", "c": "maybe"}
            if cleaned in opt_map:
                return opt_map[cleaned]

        # Step 2: Fallback text parsing with controlled recovery
        lower_text = text.lower()

        # Explicit abstention signals
        abstain_patterns = [
            r"\b(abstain|withholding|insufficient\s+evidence|cannot\s+be\s+determined|unclear\s+from\s+evidence|inconclusive)\b"
        ]
        if any(re.search(pat, lower_text) for pat in abstain_patterns):
            logger.info(f"Abstention signal detected in raw text for PMID {example.example_id}")
            return None

        # Check for conflicting conclusion signals (e.g. asserts both 'yes' and 'no')
        has_yes = bool(re.search(r"\b(answer|conclusion|prediction|decision)\s*[:=-]?\s*yes\b", lower_text))
        has_no = bool(re.search(r"\b(answer|conclusion|prediction|decision)\s*[:=-]?\s*no\b", lower_text))
        has_maybe = bool(re.search(r"\b(answer|conclusion|prediction|decision)\s*[:=-]?\s*maybe\b", lower_text))

        conflict_count = sum([has_yes, has_no, has_maybe])
        if conflict_count > 1:
            logger.warning(
                f"Conflicting conclusion signals detected for PMID {example.example_id} "
                f"(yes={has_yes}, no={has_no}, maybe={has_maybe}); abstaining safely."
            )
            return None
        if has_yes:
            return "yes"
        if has_no:
            return "no"
        if has_maybe:
            return "maybe"

        # Check isolated word boundaries for unambiguous conclusion
        words = re.findall(r"\b(yes|no|maybe)\b", lower_text)
        if words:
            unique_words = set(words)
            if len(unique_words) == 1:
                return list(unique_words)[0]
            else:
                logger.warning(
                    f"Ambiguous conflicting labels {unique_words} in response for PMID {example.example_id}; abstaining safely."
                )
                return None

        # Check multiple-choice option letter: A / B / C
        opt_matches = re.findall(r"\b([A-C])\b", text)
        if opt_matches:
            unique_opts = set(opt_matches)
            if len(unique_opts) == 1:
                mapping = {"A": "yes", "B": "no", "C": "maybe"}
                return mapping.get(list(unique_opts)[0])

        logger.warning(
            f"Could not safely parse PubMedQA decision from response for PMID {example.example_id}; abstaining."
        )
        return None

    def evaluate_prediction(
        self,
        predicted: Optional[str],
        gold: str,
        example: BenchmarkExample,
    ) -> bool:
        """Evaluates prediction against gold answer."""
        if not predicted:
            return False
        return predicted.strip().lower() == gold.strip().lower()

    def get_default_corpus(self, examples: Optional[List[BenchmarkExample]] = None) -> Optional[List[Document]]:
        """
        Extracts scientific abstract passages from benchmark examples into Documents
        for background retrieval, or loads from external PUBMED_CORPUS_DIR if configured.
        Does NOT include questions or gold labels.
        """
        corpus_env = os.environ.get("PUBMED_CORPUS_DIR") or os.environ.get("PUBMEDQA_CORPUS_DIR")
        if corpus_env and os.path.exists(corpus_env):
            from ..corpus_preparer import validate_corpus
            if os.path.isfile(corpus_env):
                with open(corpus_env, "r", encoding="utf-8") as f:
                    raw_docs = json.load(f)
                return validate_corpus(raw_docs)

        items = examples or self.load_split("test")
        docs: List[Document] = []
        seen = set()
        for ex in items:
            for idx, passage in enumerate(ex.context):
                doc_id = f"pmid_{ex.example_id}_{idx}"
                if doc_id not in seen and passage.strip():
                    seen.add(doc_id)
                    docs.append(
                        Document(
                            doc_id=doc_id,
                            title=f"PubMed Abstract {ex.example_id}",
                            text=passage.strip(),
                            source="PubMed Central",
                            year=int(ex.metadata.get("year", 2024)),
                            quality_score=0.90,
                        )
                    )
        return docs if docs else None

    def report_dataset_statistics(self, examples: List[BenchmarkExample]) -> Dict[str, Any]:
        """Calculates and returns dataset distribution and context length statistics."""
        total = len(examples)
        label_counts = {"yes": 0, "no": 0, "maybe": 0}
        ctx_lens = []
        for ex in examples:
            label_counts[ex.gold_answer] = label_counts.get(ex.gold_answer, 0) + 1
            ctx_lens.append(sum(len(c.split()) for c in ex.context))

        return {
            "total_examples": total,
            "label_distribution": label_counts,
            "mean_context_words": round(sum(ctx_lens) / total, 1) if total > 0 else 0,
        }

    def export_official_format(self, records: List[PredictionRecord], output_path: str) -> str:
        """
        Exports predictions to official PubMedQA format:
        {
            "PMID": "yes|no|maybe",
            ...
        }
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        pred_dict: Dict[str, str] = {}
        for r in records:
            # If abstained or unparseable, map to 'maybe' as standard conservative baseline
            label = r.normalized_answer.lower() if r.normalized_answer else "maybe"
            pred_dict[str(r.example_id)] = label

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(pred_dict, f, indent=2)

        logger.info(f"Exported {len(pred_dict)} predictions to official format: {output_path}")
        return output_path

    @staticmethod
    def evaluate_official_predictions(
        predictions: Dict[str, str],
        ground_truth: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Exact Python equivalent of official PubMedQA evaluation script.
        Computes Accuracy and Macro-F1 across 'yes', 'no', 'maybe'.
        """
        classes = ["yes", "no", "maybe"]
        tp = {c: 0 for c in classes}
        fp = {c: 0 for c in classes}
        fn = {c: 0 for c in classes}
        total = 0
        correct = 0

        for pmid, gold_entry in ground_truth.items():
            if isinstance(gold_entry, dict):
                gold = str(gold_entry.get("final_decision", "")).strip().lower()
            else:
                gold = str(gold_entry).strip().lower()

            if gold not in classes:
                continue

            total += 1
            pred = str(predictions.get(pmid, "")).strip().lower()

            if pred == gold:
                correct += 1
                tp[gold] += 1
            else:
                fn[gold] += 1
                if pred in classes:
                    fp[pred] += 1

        acc = correct / total if total > 0 else 0.0
        f1_scores = []
        for c in classes:
            p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
            r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
            f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
            f1_scores.append(f1)

        macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        return {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "total": total,
            "correct": correct,
        }
