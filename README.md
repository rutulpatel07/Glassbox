# Redrob — Intelligent Candidate Discovery & Ranking

This repository is the complete submission for the Redrob "India Runs"
Intelligent Candidate Discovery & Ranking hackathon (Hack2Skill). It ranks
the top 100 candidates from a 100,000-profile pool against a single Job
Description using **Cascade Variant S** — a **glass-box, fully deterministic,
zero-AI, zero-network** ranker. Every score is a transparent decomposition of
named factors; the reasoning column is a deterministic readout of the same
trace — it can never contradict the rank or hallucinate a fact the candidate
does not have. No LLM calls, no embeddings API calls, no GPU, and no network
access happen anywhere in the ranking path.

## Reproduce the submission (single command)

```bash
pip install -r requirements.txt
python3 rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

Expected composite: **0.7229** (NDCG@10 0.7964 · NDCG@50 0.7417 · MAP 0.3482 · P@10 1.0000).

Runs **CPU-only, no GPU, no network, no LLM, no model weights** on the full
100 K-candidate pool. Measured on Apple Silicon M-series (single process):
**≈ 30 s wall-clock** — well inside the 5 min budget. `candidates.jsonl` is
the organizer-provided candidate pool — it is not committed to this repo
(input data, not our code); place it in the repo root (or pass `--candidates`
with its path) before running.

Output is spec-compliant: `candidate_id,rank,score,reasoning`, exactly 100 rows,
unique ranks 1–100, monotone non-increasing score (strictly decreasing, so no
tie-break ambiguity is possible), byte-identical across runs.

### Validate the output

```bash
python validate_submission.py submission.csv
```

Checks all Section 3 rules (UTF-8, header/row count, rank uniqueness,
candidate_id format and uniqueness, score monotonicity, tie-break ordering).
Exit 0 = all pass.

### Reasoning generation

Reasoning is **not** a separate post-processing step — `rank.py` imports
`make_rich_reasoning()` directly from `reasoning.py` during ranking, so the
reasoning column is produced from the exact same score trace as the rank in
one pass. There is nothing further to run.

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

## Repository contents

Every file in this repo, grouped by role. If a file isn't listed here, it
shouldn't be in the repo — flag it.

### Core engine (the actual ranking pipeline)

| File | Role |
|---|---|
| `rank.py` | **The engine.** Single command produces the submission CSV — six scoring layers (honeypot, role-fit, domain evidence, nine fit pillars, penalties, behavioral multiplier), population calibration, Isolation Forest anomaly detection, bootstrap rank-confidence, and the Variant S two-stage selection/ordering cascade. |
| `reasoning.py` | Deterministic reasoning-string generator. Imported directly by `rank.py` (`make_rich_reasoning()`) — reads the same score trace that produced the rank, so reasoning can never contradict or hallucinate. Not run standalone in the current pipeline. |
| `requirements.txt` | Pinned dependency versions. Not boilerplate — `IsolationForest` (Phase 3) can produce different anomaly flags across scikit-learn versions even with a fixed random seed, so an unpinned environment can silently change which candidates land in the top 100. Install exactly these versions. |
| `submission_metadata.yaml` | Portal metadata mirroring what's submitted via the Hack2Skill form (team info, GitHub/sandbox links, compute environment, AI-tools declaration, methodology summary) — required by spec Section 10.2/10.3. |
| `campuscollab.csv` | The generated top-100 ranked output from `rank.py` — same content the reproduce command above produces. **Rename to your registered Hack2Skill participant ID before final upload** (spec Section 2: filename must be `<participant_id>.csv`). |

### Validation & scoring tools

| File | Role |
|---|---|
| `validate_submission.py` | Format validator — checks every Section 3 rule (UTF-8, header, row/rank counts, candidate_id uniqueness and format, score monotonicity, tie-break ordering) before you ever upload. |
| `validation_harness.py` | Internal scorer: NDCG@10/50, MAP, P@10, composite — plus `ablate` (pillar ablation), `dual` (two-scorer cross-check), and `selftest` subcommands. |
| `score_submission.py` | Scores any already-built submission CSV against `labels_filled 500.csv` without re-running the ranker — fast iteration on scoring alone. |
| `tune.py` | Tier-prediction threshold auto-fitter, used during development to calibrate tier cutoffs. Not part of the runtime ranking path. |
| `labels_filled 500.csv` | 500 hand-labeled ground-truth relevance tiers (0–5), dual-labeler reconciled, used by `validation_harness.py` / `score_submission.py` for local scoring. |

### Rejected-approach evidence (Stage 5 defense material)

These are deliberately kept, not dead weight — they document genuine
iteration and dead ends, which the evaluation pipeline explicitly checks for
at Stage 4/5 (git history authenticity, ability to defend design choices).

| File | Role |
|---|---|
| `experiments.py` | Variant builders A–W: the formal LambdaMART/gradient-boosted learning-to-rank bake-off (rejected — collapsed on the real scorer despite a strong local validation number), the semantic-embeddings experiment (rejected — RAM/latency budget), pairwise reranking (v5, rejected), behavioral-multiplier neutralization (v4, rejected), and the two intermediate cascade widths (Cascade T/U) that preceded the correct 1–15 head/tail split in Variant B. |
| `ml_engine.py` | An earlier, independently-built candidate ranker (TF-IDF vectorization + cosine similarity + a Gaussian bell curve) from before this project's deterministic-lexicon architecture was adopted. Kept as an archived contrast to `rank.py`, not part of the submission pipeline. |
| `main.py` | CLI entry point for `ml_engine.py` above. Not used by `rank.py`; kept alongside it as the same archived reference. |
| `AUDIT.md` | A full architecture audit snapshot from an earlier point in development (pre-Cascade-Variant-S, submission scored 0.6421 at the time). Useful as evidence of an early rigor pass; note the specific numbers/file tree it describes are historical, not the current state — see the Scores table below for current numbers. |
| `EXPLAINER_for_Rutul.md` | Plain-language walkthrough of the engine's design decisions, written for internal team onboarding. |

### Reproducibility / demo

| File | Role |
|---|---|
| `sandbox.ipynb` | Colab notebook implementing the ranker end-to-end for the hosted sandbox requirement (spec Section 10.5) — accepts a small candidate sample and produces a ranked CSV within the compute budget. |

`rank.py --report` generates a self-contained HTML report (pillar waterfall
charts, evidence pills, honeypot flags) from an already-ranked top-100 — run
it locally if you want one; no example is committed to the repo.

### Not in this repo (by design)

`candidates.jsonl` (the 100K candidate pool) and the full organizer hackathon
bundle are excluded via `.gitignore` — they're organizer-provided input data,
not our code, and `candidates.jsonl` alone is too large for a normal git push.
Place it locally before running `rank.py`.

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
