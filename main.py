"""
MedRag — Main Entry Point
Adaptive Evidence-Aware Retrieval and Verification Framework
for Reliable Medical Question Answering

Usage examples
──────────────
  # Interactive mode (mock LLM, built-in corpus):
  python main.py

  # Single question with OpenAI:
  python main.py -q "Is metformin safe for patients with renal impairment?"

  # Demo questions with Groq:
  python main.py --demo --backend groq --model llama-3.3-70b-versatile

  # Custom corpus + max rounds:
  python main.py --corpus data/pubmed.jsonl --max-rounds 4 -q "..."

  # Disable reranker for faster (less accurate) retrieval:
  python main.py --no-reranker --demo
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# ── Sample corpus ─────────────────────────────────────────────────────────────

SAMPLE_CORPUS_DATA = [
    {
        "id": "1", "year": 2024, "quality_score": 0.95, "source": "NEJM",
        "title": "Metformin as First-Line Therapy for Type 2 Diabetes",
        "text": (
            "Metformin is the recommended first-line pharmacological agent for type 2 "
            "diabetes mellitus. It reduces HbA1c by 1–2% and has a favourable safety "
            "profile. Contraindicated in severe renal impairment (eGFR < 30 mL/min/1.73 m²). "
            "At eGFR 30–45, use with caution and reduce dose. Associated with lactic "
            "acidosis risk when renal clearance is impaired."
        ),
    },
    {
        "id": "2", "year": 2023, "quality_score": 0.93, "source": "Lancet",
        "title": "Metformin Safety in Chronic Kidney Disease",
        "text": (
            "Metformin is absolutely contraindicated when eGFR falls below "
            "30 mL/min/1.73 m². Between eGFR 30–45 use with dose reduction and close "
            "monitoring. The risk of lactic acidosis is rare but potentially fatal. "
            "Regular renal function monitoring is mandatory. Temporary cessation is "
            "advised before contrast procedures or major surgery."
        ),
    },
    {
        "id": "3", "year": 2025, "quality_score": 0.94, "source": "NEJM",
        "title": "SGLT2 Inhibitors and Cardiovascular Outcomes",
        "text": (
            "SGLT2 inhibitors (empagliflozin, dapagliflozin, canagliflozin) provide "
            "cardiovascular and renal protection in type 2 diabetes. Empagliflozin reduced "
            "cardiovascular death by 38% in the EMPA-REG OUTCOME trial. Recommended for "
            "patients with established cardiovascular disease or high cardiovascular risk. "
            "Risk of genitourinary infections and DKA should be monitored."
        ),
    },
    {
        "id": "4", "year": 2024, "quality_score": 0.91, "source": "BMJ",
        "title": "GLP-1 Receptor Agonists: Efficacy and Safety",
        "text": (
            "GLP-1 receptor agonists (semaglutide, liraglutide, dulaglutide) reduce HbA1c "
            "by 1–2% and promote significant weight loss (5–15 kg). The LEADER trial "
            "demonstrated liraglutide's cardiovascular benefit. Primary side effects are "
            "nausea, vomiting, and diarrhoea. Contraindicated in personal or family history "
            "of medullary thyroid carcinoma or MEN2."
        ),
    },
    {
        "id": "5", "year": 2023, "quality_score": 0.90, "source": "ADA Standards",
        "title": "ADA Standards of Medical Care in Diabetes 2023",
        "text": (
            "ADA 2023 guidelines recommend metformin as initial therapy unless contraindicated. "
            "For patients with established ASCVD, heart failure, or CKD, add an SGLT2 inhibitor "
            "or GLP-1 agonist regardless of HbA1c. Insulin remains the most potent "
            "glucose-lowering agent and is needed when HbA1c > 10% or hyperglycaemic symptoms "
            "are present. Individualise treatment based on comorbidities and patient preference."
        ),
    },
    {
        "id": "6", "year": 2022, "quality_score": 0.89, "source": "USPSTF",
        "title": "Aspirin Use for Primary Prevention of Cardiovascular Disease",
        "text": (
            "The USPSTF 2022 recommendation advises against initiating low-dose aspirin for "
            "primary prevention of CVD in adults aged 60 years or older. For adults aged 40–59 "
            "with ≥10% 10-year CVD risk, the decision should be individualised. Benefits of "
            "aspirin for primary prevention do not outweigh harms (mainly GI bleeding). "
            "Aspirin remains indicated for secondary prevention after MI or stroke."
        ),
    },
    {
        "id": "7", "year": 2024, "quality_score": 0.92, "source": "ESC Guidelines",
        "title": "Anticoagulation in Non-Valvular Atrial Fibrillation",
        "text": (
            "Direct oral anticoagulants (DOACs: apixaban, rivaroxaban, dabigatran, edoxaban) "
            "are preferred over warfarin for stroke prevention in non-valvular AF. DOACs "
            "demonstrate similar or superior efficacy with lower risk of intracranial "
            "haemorrhage. No INR monitoring required. Warfarin remains indicated for AF with "
            "mechanical heart valves or moderate-to-severe mitral stenosis."
        ),
    },
    {
        "id": "8", "year": 2024, "quality_score": 0.93, "source": "AHA/ACC",
        "title": "Heart Failure with Reduced Ejection Fraction: GDMT",
        "text": (
            "Guideline-directed medical therapy (GDMT) for HFrEF includes ACE inhibitors/ARBs "
            "(or ARNI), beta-blockers, MRA, and SGLT2 inhibitors. ARNI (sacubitril/valsartan) "
            "reduces cardiovascular death and HF hospitalisations vs enalapril (PARADIGM-HF). "
            "Ivabradine added if HR ≥ 70 bpm on maximally tolerated beta-blocker. "
            "Target: maximally tolerated doses of all four pillars."
        ),
    },
    {
        "id": "9", "year": 2023, "quality_score": 0.88, "source": "IDSA",
        "title": "Antibiotic Management of Streptococcal Pharyngitis",
        "text": (
            "Penicillin V or amoxicillin (10-day course) is first-line therapy for Group A "
            "streptococcal pharyngitis. Amoxicillin is preferred in children for palatability. "
            "In penicillin allergy: azithromycin (5 days) or clindamycin (10 days). Cephalexin "
            "may be used for non-anaphylactic penicillin allergy. Antibiotic stewardship "
            "requires confirmed strep test (RADT or culture) before prescribing."
        ),
    },
    {
        "id": "10", "year": 2023, "quality_score": 0.91, "source": "ACC/AHA",
        "title": "Statin Therapy for Primary and Secondary Prevention",
        "text": (
            "High-intensity statins (atorvastatin 40–80 mg, rosuvastatin 20–40 mg) are "
            "recommended for secondary prevention after ASCVD. For primary prevention, "
            "statin use is guided by 10-year ASCVD risk (≥7.5%). LDL reduction ≥50% is the "
            "primary target. Statin-associated muscle symptoms occur in 5–10% of patients; "
            "rhabdomyolysis is rare (<0.1%). Annual CK and LFT monitoring is no longer "
            "routinely recommended."
        ),
    },
    {
        "id": "11", "year": 2024, "quality_score": 0.91, "source": "Critical Care Med",
        "title": "Surviving Sepsis Campaign: 2024 Update",
        "text": (
            "Sepsis management requires hour-1 bundle: blood cultures before antibiotics, "
            "broad-spectrum IV antibiotics within 1 hour, 30 mL/kg IV crystalloid for "
            "hypotension, vasopressors for MAP < 65 mmHg (noradrenaline first-line), "
            "and lactate measurement. De-escalate antibiotics when cultures return. "
            "Steroid use (hydrocortisone 200 mg/day) in septic shock refractory to vasopressors."
        ),
    },
    {
        "id": "12", "year": 2023, "quality_score": 0.90, "source": "AHA/ACC",
        "title": "Hypertension Management — 2023 Clinical Guidelines",
        "text": (
            "Target BP is < 130/80 mmHg for most adults. First-line antihypertensives: "
            "ACE inhibitors, ARBs, calcium channel blockers, and thiazide diuretics. "
            "ACE inhibitors/ARBs preferred in diabetes with CKD or proteinuria. "
            "Calcium channel blockers preferred in Black patients. Beta-blockers are "
            "second-line unless a compelling indication exists (HF, post-MI, tachyarrhythmia)."
        ),
    },
    {
        "id": "13", "year": 2025, "quality_score": 0.89, "source": "Chest",
        "title": "Venous Thromboembolism Treatment and Secondary Prevention",
        "text": (
            "DOACs are preferred over LMWH/warfarin for most patients with DVT/PE. "
            "Apixaban and rivaroxaban can be started immediately without parenteral bridging. "
            "Duration: provoked VTE 3 months; unprovoked VTE at least 3 months, consider "
            "indefinite if low bleeding risk. Cancer-associated VTE: LMWH or DOACs "
            "(edoxaban/rivaroxaban) preferred. Thrombolysis for massive PE with haemodynamic "
            "compromise."
        ),
    },
    {
        "id": "14", "year": 2022, "quality_score": 0.87, "source": "AASLD",
        "title": "Management of Non-Alcoholic Fatty Liver Disease",
        "text": (
            "Lifestyle intervention (weight loss ≥7–10%) is the cornerstone of NAFLD "
            "management. No drug is currently FDA-approved specifically for NAFLD/NASH. "
            "Vitamin E (800 IU/day) is recommended in non-diabetic adults with biopsy-proven "
            "NASH. Pioglitazone improves histology but causes weight gain. Semaglutide shows "
            "promise in Phase 3 trials. Avoid hepatotoxic drugs; screen for cirrhosis."
        ),
    },
    {
        "id": "15", "year": 2024, "quality_score": 0.90, "source": "GOLD",
        "title": "COPD: GOLD 2024 Management Guidelines",
        "text": (
            "GOLD 2024 classifies COPD by symptoms (mMRC/CAT) and exacerbation history. "
            "All patients: smoking cessation and bronchodilators. Group A: SABA or SAMA. "
            "Group B: LABA + LAMA (dual therapy). Group E: LABA + LAMA ± ICS if "
            "eosinophils ≥300 or frequent exacerbations. Pulmonary rehabilitation "
            "improves exercise capacity and quality of life. Long-term oxygen for "
            "PaO2 ≤ 55 mmHg."
        ),
    },
]


def _build_sample_corpus():
    from medrag import Document
    return [
        Document(
            doc_id=d["id"],
            title=d["title"],
            text=d["text"],
            source=d["source"],
            year=d["year"],
            quality_score=d["quality_score"],
        )
        for d in SAMPLE_CORPUS_DATA
    ]


def load_corpus_file(path: str):
    from medrag import Document
    docs = []
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        items = [json.loads(l) for l in lines if l.strip()]
    else:
        with open(path, encoding="utf-8") as f:
            items = json.load(f)
    if not isinstance(items, list):
        items = [items]
    for i, d in enumerate(items):
        docs.append(Document(
            doc_id=str(d.get("id", i)),
            title=d.get("title", ""),
            text=d.get("text", d.get("abstract", d.get("content", ""))),
            source=d.get("source", d.get("journal", "")),
            year=int(d.get("year", d.get("pub_year", 2020))),
            quality_score=float(d.get("quality_score", 0.5)),
        ))
    return docs


# ── Demo questions ────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    "What is the first-line treatment for type 2 diabetes, and is metformin safe for patients with renal impairment?",
    "Should aspirin be used for primary prevention of cardiovascular events in elderly patients?",
    "What anticoagulant is preferred for stroke prevention in non-valvular atrial fibrillation?",
]


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MedRag: Adaptive Evidence-Aware Medical QA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("-q", "--question", help="Medical question to answer")
    p.add_argument("-c", "--corpus", help="Path to corpus JSON or JSONL file")
    p.add_argument(
        "-b", "--backend",
        default="auto",
        choices=["openai", "groq", "ollama", "mock", "auto"],
        help="LLM backend (default: auto-detect from env vars)",
    )
    p.add_argument("-m", "--model", help="LLM model name override")
    p.add_argument("-k", "--api-key", help="LLM API key override")
    p.add_argument("--base-url", help="LLM API base URL (for Groq/Ollama)")
    p.add_argument("--max-rounds", type=int, default=3, help="Max retrieval rounds (default 3)")
    p.add_argument("--no-reranker", action="store_true", help="Disable cross-encoder reranker")
    p.add_argument("--demo", action="store_true", help="Run built-in demo questions")
    p.add_argument("--output", help="Save result JSON to this path")
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    try:
        from config import MedRagConfig
        config = MedRagConfig.from_env()
    except ImportError:
        from medrag.pipeline import _DefaultConfig
        config = _DefaultConfig()

    config.max_search_rounds = args.max_rounds
    if args.no_reranker:
        config.use_reranker = False

    # ── LLM backend ───────────────────────────────────────────────────────────
    from medrag import LLMClient
    backend = args.backend
    model = args.model
    api_key = args.api_key
    base_url = args.base_url

    if backend == "auto":
        if os.environ.get("OPENAI_API_KEY"):
            backend, model = "openai", model or "gpt-4o-mini"
        elif os.environ.get("GROQ_API_KEY"):
            backend = "groq"
            api_key = api_key or os.environ.get("GROQ_API_KEY")
            base_url = base_url or "https://api.groq.com/openai/v1"
            model = model or "llama-3.3-70b-versatile"
        else:
            backend = "mock"
            print("[Info] No LLM API key found — using mock backend (for testing).")
            print("[Info] Set OPENAI_API_KEY or GROQ_API_KEY for real answers.\n")

    llm = LLMClient(
        backend=backend,
        model=model or getattr(config, "llm_model", "gpt-4o-mini"),
        api_key=api_key or getattr(config, "llm_api_key", None),
        base_url=base_url or getattr(config, "llm_base_url", None),
    )

    # ── Corpus ────────────────────────────────────────────────────────────────
    if args.corpus:
        print(f"Loading corpus from {args.corpus} ...")
        documents = load_corpus_file(args.corpus)
    else:
        documents = _build_sample_corpus()
        print(f"Using built-in sample corpus ({len(documents)} documents)")

    # ── Pipeline ──────────────────────────────────────────────────────────────
    from medrag import MedRagPipeline
    pipeline = MedRagPipeline(documents=documents, llm_client=llm, config=config)

    results = []

    if args.demo:
        for q in DEMO_QUESTIONS:
            r = pipeline.answer(q)
            r.print_summary()
            results.append(r.to_dict())

    elif args.question:
        r = pipeline.answer(args.question)
        r.print_summary()
        results.append(r.to_dict())

    else:
        # Interactive
        print("\nMedRag Interactive Mode  (type 'quit' to exit)\n")
        while True:
            try:
                q = input("Question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q:
                continue
            if q.lower() in ("quit", "exit", "q"):
                break
            r = pipeline.answer(q)
            r.print_summary()
            results.append(r.to_dict())

    # ── Optional output ───────────────────────────────────────────────────────
    if args.output and results:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
