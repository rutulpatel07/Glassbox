# Redrob — Intelligent Candidate Discovery & Ranking

A **glass-box, fully deterministic, zero-AI, zero-network** candidate ranker
built for the Redrob hackathon. Every score is a transparent decomposition of
named factors; the reasoning column is a deterministic readout of the same
trace — it can never contradict the rank or hallucinate a fact the candidate
does not have.

## Reproduce the submission (single command)

```bash
pip install -r requirements.txt
python3 rank.py --candidates ./candidates.jsonl --out ./submission_final.csv
```

Expected composite: **0.7229** (NDCG@10 0.7964 · NDCG@50 0.7417 · MAP 0.3482 · P@10 1.0000).

Runs **CPU-only, no GPU, no network, no LLM, no model weights** on the full
100 K-candidate pool. Measured on Apple Silicon M-series (single process):
**≈ 30 s wall-clock** — well inside the 5 min budget.

Output is spec-compliant: `candidate_id,rank,score,reasoning`, exactly 100 rows,
unique ranks 1–100, monotone non-increasing score, byte-identical across runs.

### Validate the output

```bash
python validate_submission.py submission.csv
```

Checks all Section 3 rules (UTF-8, row count, rank uniqueness, candidate ID
pool membership, score monotonicity, non-empty reasoning). Exit 0 = all pass.

### Generate richer reasoning (optional)

```bash
python reasoning.py   # reads submission_v3point2.csv, writes submission_v3point2_with_reasoning.csv
```

Produces pillar-specific, employer-named, concern-surfacing 1–2-sentence
reasoning strings from the same score trace. Zero LLM calls.

## Methodology

The ranker is a five-phase deterministic pipeline. **Phase 1** extracts raw
features for every candidate via six scoring layers: L0 honeypot detection
(flags impossible profiles with internal contradictions — expert skills claimed
for zero months, tenure/date mismatches, multiple `is_current` jobs); L1
role-fit gate (floors non-engineering and non-target-tech titles regardless of
keyword density); L2 domain evidence (reads `career_history.description` for
explicit retrieval/LTR/recsys/vector/eval signal, hard-capped for plain-ML
adjacency to avoid inflating filler candidates); L3 fit pillars (nine weighted
components: domain evidence, skill substance, seniority, product-vs-services
background, external validation, eval-framework literacy, semantic similarity,
Python signal, location, notice, platform quality); L4 do-NOT-want penalties
(research-only, IT-services-only, CV/robotics-heavy, job-hopping — applied as
multiplicative deductions not floors); and L5 behavioral availability (recency,
response rate, interview completion, open-to-work flag, profile completeness).
**Phase 2** population-calibrates the three most distribution-sensitive pillars
(domain evidence, skill substance, eval frameworks) to percentile ranks across
the full 100 K pool. **Phase 3** runs an Isolation Forest on 10 numeric features
to catch anomalous profiles the rule-based gate missed. **Phase 4** computes a
rank-stability diagnostic by perturbing weights 500 times and recording the
fraction of configurations in which each candidate lands in the top 100; this
is a sensitivity-to-weight-perturbation diagnostic appended to the reasoning
column for transparency — it does not influence the ranking.
**Phase 5** implements the Variant S two-stage cascade: the pipeline decomposes
selection and ordering into separate concerns. Stage 1 selects the top-100
candidates by `final_select` (domain signal without behavioral noise). The head
(positions 1–15) is drawn from the top-15 by `final_order` and then reordered
by `final_order × (0.85 + 0.15 × t5)`, where t5 is a platform engagement
signal (`0.35·offer_acceptance + 0.25·search_appearance + 0.20·saved_by_recruiters
+ 0.20·assessment_score`) — surfacing high-intent candidates into visible slots.
The tail (positions 16–100) is ordered by calibrated `domain_evidence` score
descending, ensuring the bulk of the list tracks subject-matter depth rather
than recency of login. A monotone score cap enforces spec compliance.

## Pipeline files

| File | Role |
|---|---|
| `rank.py` | Main engine — single command produces the submission CSV |
| `reasoning.py` | Richer deterministic reasoning layer (optional post-step) |
| `validate_submission.py` | Spec-compliance checker (Section 3 rules) |
| `validation_harness.py` | NDCG/MAP scorer, ablation, dual-scorer, self-test |
| `score_submission.py` | Score any pre-built submission against labels |
| `train_ltr.py` | Optional LambdaMART re-ranker (not used in final submission) |
| `requirements.txt` | Pinned dependencies |
| `submission_metadata.yaml` | Portal metadata |
| `candidates.jsonl` | 100 K candidate pool (input, not committed to repo) |
| `labels_filled 500.csv` | 500 hand-labeled ground-truth tiers |

## Validation harness

```bash
python validation_harness.py score   --candidates candidates.jsonl --labels "labels_filled 500.csv"
python validation_harness.py ablate  --candidates candidates.jsonl
python validation_harness.py dual    --candidates candidates.jsonl
python validation_harness.py selftest --candidates candidates.jsonl
```

## Scores (local, against 500 labels)

| Submission | NDCG@10 | NDCG@50 | MAP | P@10 | **COMPOSITE** |
|---|---|---|---|---|---|
| v3\_calibrated\_v2 (prev best) | 0.7601 | 0.6020 | 0.2096 | 1.0000 | 0.6421 |
| v3point2\_with\_reasoning | 0.7699 | 0.6019 | 0.2511 | 1.0000 | 0.6532 |
| **submission\_final (Variant S)** | **0.7964** | **0.7417** | **0.3482** | **1.0000** | **0.7229** |
