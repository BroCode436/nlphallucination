# Adaptive Evidence-Guided Hallucination Detection and Correction with Post-Correction Verification for Large Language Models

**Course Evaluation:** MSE-2 (College Research Project)  
**Academic Level:** 2nd-Year B.Tech CSE (Artificial Intelligence)  
**Primary Dataset:** RAGTruth (17,790 LLM responses across QA, Summary, Data2txt)  
**Scientific Standards:** IEEE Peer-Review Standard (Zero Fabricated Metrics, Empirical Wall-Clock Latency, Zero Source-ID Leakage)

---

## 1. Project Overview & Research Question

Large Language Models (LLMs) used in Retrieval-Augmented Generation (RAG) often produce hallucinations—factual conflicts or ungrounded baseless claims. 

### Core Research Question:
> *"Can adaptive evidence-guided verification maintain hallucination detection quality while drastically reducing unnecessary verification effort compared to uniform fixed verification?"*

### Proposed Pipeline:
```
LLM Response
      ↓
Claim Extraction (Sentence-level spans)
      ↓
Evidence Quality Assessment (Strong, Weak, Irrelevant, Absent)
      ↓
Adaptive Tiered Verification
  ├── Fast Path: High confidence + Strong evidence → 1 verifier → Accept
  ├── Medium Path: Ambiguous / Borderline → 2 verifiers consensus
  ├── Deep Path: Low confidence / Conflict → 3 verifiers + Targeted Correction
  └── Abstention: No reliable evidence → Abstain ("Not Verifiable")
      ↓
Targeted Claim-Level Correction (Evidence-grounded aligned substitution)
      ↓
Post-Correction Re-verification (Ensures no new hallucinations introduced)
      ↓
Final Verified Output
```

---

## 2. Directory Layout

```
hallucination_project/
├── AUDIT_REPORT.md                    # Comprehensive scientific & technical code audit
├── RESULTS_SUMMARY.md                 # Empirical findings, sample sizes, and limitations
├── README.md                          # Reproduction guide and execution commands
├── requirements.txt                   # Environment package dependencies
├── run_pipeline.py                    # Unified CLI pipeline entry point
│
├── configs/
│   └── config.json                    # Centralized hyperparameters, seeds, and paths
│
├── tests/                             # Automated unit & integration test suite
│   ├── test_preprocessing.py          # Sentence splitting & zero-leakage assertions
│   ├── test_verification.py           # Evidence quality, conflict rules & abstention
│   ├── test_correction.py             # Targeted repairs, CCPR, CIHR, and metrics
│   └── test_pipeline.py               # End-to-end integration smoke test
│
├── data/
│   ├── raw/                           # Raw RAGTruth dataset (response.jsonl, source_info.jsonl)
│   └── processed/                     # Leak-free claims (96k train, 17k val, 18k test)
│
├── notebooks/
│   ├── ragtruth_mse1.ipynb            # MSE-1: EDA, Preprocessing, Baseline formulation
│   └── ragtruth_mse2.ipynb            # MSE-2: Research-grade verification & correction
│
├── src/
│   ├── preprocessing.py               # Raw dataset loader & claim extractor
│   ├── train.py                       # Baseline & Transformer detector training
│   ├── verification.py                # Verification engine & adaptive routing policy
│   ├── correction.py                  # Evidence-guided correction & operational metrics
│   └── evaluate.py                    # Master benchmark, dual distribution evaluation, ablations
│
├── results/
│   ├── checkpoints/                   # Fitted Logistic Regression & DistilBERT weights
│   ├── figures/                       # Publication-ready plots (F1 tradeoff, confusion matrix)
│   ├── tables/                        # CSV benchmark tables, ablations, and subtype metrics
│   └── predictions/                   # Granular per-claim correction and evaluation logs
│
└── paper/
    └── ieee_research_paper.md         # Full IEEE-formatted research paper draft
```

---

## 3. How to Run & Reproduce

### Quick Start (Run Entire Pipeline & Tests)
```bash
py -3.12 run_pipeline.py --all  # (Options: --test, --preprocess, --train, --force-train, --evaluate)
```

### Run Automated Test Suite Only
```bash
py -3.12 -m unittest discover tests
```

### Run Data Preprocessing (with Leakage Assertions)
```bash
py -3.12 src/preprocessing.py
```

### Run Baseline & Transformer Training
```bash
py -3.12 src/train.py
```

### Run Evaluation Suite (Dual Distribution & Latency Profiling)
```bash
py -3.12 src/evaluate.py
```

### Update the Jupyter Notebook
```bash
py -3.12 build_updated_mse2_notebook.py
```

---

## 4. Key Verified Experimental Findings

1. **Zero Data Leakage:** Train (1,417 sources), Val (250 sources), and Test (295 sources) have strictly **0 overlapping source IDs**.
2. **Dual Distribution Reality:**
   - On the balanced test subset (50/50), DistilBERT cross-encoder achieves **F1 = 0.6701** (Accuracy 0.6790) vs. TF-IDF baseline **F1 = 0.5782**.
   - On the natural test distribution (~8.7% prevalence), DistilBERT achieves **F1 = 0.2638** vs. TF-IDF baseline **F1 = 0.2169**, reflecting real-world class imbalance.
3. **Adaptive Verification Efficiency & Precision:**
   - Adaptive verification achieves **0.6571 F1**, **0.8050 Precision**, and **0.6968 Accuracy** on covered claims (92.33% coverage, abstaining on 46 ambiguous cases), reducing verification overhead from 3.00 calls down to 2.45 calls per claim.
4. **Correction Fidelity & Safety:**
   - Targeted evidence-guided correction achieves a **37.98% Correction Success Rate (CSR)** while maintaining a strict **0.00% Correction-Induced Hallucination Rate (CIHR)** and **87.50% Content Preservation (CCPR)** on factual assertions, with 14.75% explicit abstention on ambiguous context.

