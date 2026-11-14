"""
Synthetic Medical QA Dataset Adapter.
Provides deterministic, self-contained clinical vignettes with ground-truth literature
passages for reproducible offline unit testing, CI verification, and smoke runs.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..adapter import DatasetAdapter
from ..registry import BenchmarkRegistry
from ..schema import BenchmarkExample
from ...retriever import Document


# Canonical synthetic medical QA items
_SYNTHETIC_DATA: List[Dict[str, Any]] = [
    {
        "id": "syn_001",
        "question": "Which of the following antidiabetic agents is contraindicated in patients with severe renal impairment (eGFR < 30 mL/min/1.73 m2) due to the risk of lactic acidosis?",
        "options": {
            "A": "Metformin",
            "B": "Empagliflozin",
            "C": "Sitagliptin",
            "D": "Glipizide",
        },
        "gold": "A",
        "context": [
            "Metformin is an oral biguanide widely prescribed as first-line therapy for type 2 diabetes. However, metformin is strictly contraindicated in patients with advanced renal failure (eGFR < 30 mL/min/1.73 m2) due to marked accumulation and the elevated risk of life-threatening metformin-associated lactic acidosis (MALA)."
        ],
        "metadata": {"specialty": "Endocrinology / Nephrology", "difficulty": "Medium"},
    },
    {
        "id": "syn_002",
        "question": "A 58-year-old male with acute anterior myocardial infarction is stabilized. Which medication should be initiated within 24 hours to reduce long-term cardiac remodeling and mortality, provided there is no hypotension or cardiogenic shock?",
        "options": {
            "A": "Diltiazem",
            "B": "Lisinopril",
            "C": "Digoxin",
            "D": "Furosemide monotherapy",
        },
        "gold": "B",
        "context": [
            "Clinical practice guidelines for acute myocardial infarction strongly recommend starting an ACE inhibitor such as lisinopril or captopril within the first 24 hours of hospital admission in hemodynamically stable patients. ACE inhibition prevents adverse ventricular remodeling and significantly reduces post-infarct cardiovascular mortality."
        ],
        "metadata": {"specialty": "Cardiology", "difficulty": "Medium"},
    },
    {
        "id": "syn_003",
        "question": "Which direct oral anticoagulant (DOAC) is primarily cleared by the kidneys (80% renal clearance) and requires strict dose adjustment or avoidance in severe renal failure?",
        "options": {
            "A": "Apixaban",
            "B": "Rivaroxaban",
            "C": "Dabigatran",
            "D": "Edoxaban",
        },
        "gold": "C",
        "context": [
            "Dabigatran etexilate is an oral direct thrombin inhibitor. Approximately 80% of dabigatran is excreted unchanged by the kidneys, contrasting with apixaban (approx. 27% renal clearance). Consequently, severe renal impairment substantially increases dabigatran plasma concentrations and major bleeding hazard."
        ],
        "metadata": {"specialty": "Hematology / Pharmacology", "difficulty": "Hard"},
    },
    {
        "id": "syn_004",
        "question": "A 32-year-old female presents with acute severe headache, fever, and neck stiffness. Cerebrospinal fluid analysis reveals elevated opening pressure, neutrophilic pleocytosis, high protein, and markedly decreased glucose. What is the most appropriate initial empiric antimicrobial therapy?",
        "options": {
            "A": "Oral amoxicillin and clavulanate",
            "B": "Intravenous ceftriaxone, vancomycin, and ampicillin (if Listeria risk present)",
            "C": "Intravenous acyclovir monotherapy",
            "D": "Intravenous fluconazole monotherapy",
        },
        "gold": "B",
        "context": [
            "Acute bacterial meningitis is a medical emergency presenting with fever, headache, meningismus, and CSF showing marked polymorphonuclear pleocytosis with low glucose. Empiric therapy should begin immediately with intravenous third-generation cephalosporin (ceftriaxone or cefotaxime) plus vancomycin, with ampicillin added in patients aged >50 or immunocompromised to cover Listeria monocytogenes."
        ],
        "metadata": {"specialty": "Infectious Disease", "difficulty": "Medium"},
    },
    {
        "id": "syn_005",
        "question": "What is the recommended target international normalized ratio (INR) for a patient with a mechanical mitral valve prosthesis receiving warfarin anticoagulation?",
        "options": {
            "A": "1.5 - 2.0",
            "B": "2.0 - 3.0",
            "C": "2.5 - 3.5",
            "D": "3.5 - 4.5",
        },
        "gold": "C",
        "context": [
            "Mechanical heart valve prostheses carry a high thromboembolic risk. Because mechanical mitral valves have higher thrombogenicity than aortic prostheses, current cardiology guidelines recommend a target INR of 3.0 (acceptable range 2.5 to 3.5) in combination with low-dose aspirin where appropriate."
        ],
        "metadata": {"specialty": "Cardiology", "difficulty": "Hard"},
    },
]


@BenchmarkRegistry.register_adapter("synthetic")
class SyntheticAdapter(DatasetAdapter):
    """
    Adapter for deterministic synthetic clinical evaluations.
    """

    @property
    def dataset_id(self) -> str:
        return "synthetic"

    @property
    def version(self) -> str:
        return "1.0.0-fixtures"

    @property
    def supported_splits(self) -> List[str]:
        return ["dev", "val", "test", "train"]

    def load_split(self, split: str, limit: Optional[int] = None) -> List[BenchmarkExample]:
        """Loads synthetic examples for testing."""
        examples = [self.normalize_example(item, split) for item in _SYNTHETIC_DATA]
        if limit is not None and limit > 0:
            examples = examples[:limit]
        return examples

    def normalize_example(self, raw_item: Dict[str, Any], split: str) -> BenchmarkExample:
        """Converts raw fixture to BenchmarkExample."""
        return BenchmarkExample(
            example_id=str(raw_item["id"]),
            dataset=self.dataset_id,
            split=split,
            question=raw_item["question"],
            options=raw_item.get("options", {}),
            context=raw_item.get("context", []),
            gold_answer=self.extract_gold_answer(raw_item),
            metadata=raw_item.get("metadata", {}),
        )

    def extract_gold_answer(self, raw_item: Dict[str, Any]) -> str:
        return str(raw_item.get("gold", raw_item.get("answer", ""))).strip().upper()

    def normalize_answer(self, raw_response: str, example: BenchmarkExample) -> Optional[str]:
        """
        Extracts option letter (A, B, C, D) from raw model output.
        """
        if not raw_response or not raw_response.strip():
            return None

        text = raw_response.strip()

        # Check for explicit abstention signals
        if "withholding" in text.lower() or "abstain" in text.lower() or "insufficient evidence" in text.lower():
            return None

        # Look for explicit letter markers
        patterns = [
            r"(?:answer|option|choice)\s*(?:is|:)?\s*[\*\_]*([A-D])[\*\_]*\b",
            r"^[\*\_]*([A-D])[\*\_]*[\.\:\)]",
            r"\b([A-D])\s*[\.\:\)]",
            r"\b([A-D])\b",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(1).upper()

        # Check if option text appears explicitly in answer
        for opt_key, opt_text in example.options.items():
            if opt_text.lower() in text.lower():
                return opt_key.upper()

        return None

    def get_default_corpus(self, examples: Optional[List[BenchmarkExample]] = None) -> Optional[List[Document]]:
        """Extracts ground-truth passages into Documents."""
        items = examples or self.load_split("test")
        docs: List[Document] = []
        seen = set()
        for ex in items:
            for idx, passage in enumerate(ex.context):
                doc_id = f"doc_{ex.example_id}_{idx}"
                if doc_id not in seen:
                    seen.add(doc_id)
                    docs.append(Document(
                        doc_id=doc_id,
                        title=f"Clinical Evidence for {ex.example_id}",
                        text=passage,
                        source="Synthetic Curated Medical Literature",
                        year=2024,
                        quality_score=0.95,
                    ))
        return docs
