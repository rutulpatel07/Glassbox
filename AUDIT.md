# Redrob Ranking Engine — Full Audit

_Generated 2026-06-08. No code was changed during this audit._

---

## 1. File Tree

```
redrob_pkg/
├── candidates.jsonl              # 100,000-candidate pool (input)
├── rank.py                       # ★ Main ranking engine (5-phase pipeline)
├── validation_harness.py         # NDCG/MAP scorer, ablation, dual-scorer, selftest
├── score_submission.py           # Score any submission CSV against labels
├── train_ltr.py                  # Optional LambdaMART re-ranker
├── tune.py                       # Auto-fit tier-prediction thresholds (cosmetic)
├── labels_filled 500.csv         # 500 hand-labeled ground-truth tiers
├── submission_v3_calibrated_v2.csv  # Current best submission (DO NOT OVERWRITE)
├── submission_metadata.yaml      # Portal metadata (team info, methodology)
├── requirements.txt              # pip dependencies
├── README.md                     # Reproduce instructions
├── DEEPDIVE_ML_Concepts.md       # Reference notes (not in ranking path)
└── EXPLAINER_for_Rutul.md        # Reference notes (not in ranking path)
```

---

## 2. rank.py — End-to-End

### Entry command

```bash
python rank.py --candidates candidates.jsonl --out submission.csv [--top 100] [--dump-pillars features.csv]
```

### Inputs
- `candidates.jsonl` — 100K JSONL records (see §5 for full schema)
- No network calls, no model weights, no GPU

### Pipeline (five sequential phases)

#### Phase 1 — Multiprocessing feature extraction
- Splits candidates across all CPU cores via `multiprocessing.Pool`
- Each candidate runs through six scoring layers (L0–L5):

| Layer | Gate / Function | Effect |
|---|---|---|
| **L0 Honeypot** | `honeypot_flags()` | Detects impossible profiles: ≥3 expert skills with 0-month duration; skill duration > career; tenure/date contradictions; multiple `is_current` jobs. Two strong flags OR ≥2 any flags → `is_hp=True` |
| **L1 Role-fit** | `role_fit()` | Returns a `role_mult` ∈ {0.05, 0.12, 0.58, 1.00} based on current title regex. Non-technical → 0.05; non-target tech → 0.12; adjacent engineer → 0.58 (boosted by domain evidence); core ML → 1.00 |
| **L2 Domain evidence** | `domain_evidence()` | Reads full career text. Signals: retrieval/ranking terms (RX_RETRIEVAL), LLM/embedding terms (RX_LLM_MODERN), vector DB terms (RX_VECTOR_DB), eval metrics (RX_EVAL), NLP/IR (RX_NLP_IR), adjacent ML (RX_ADJACENT_ML). Explicit signal (any retrieval/LLM/vdb/eval hit): `score = clip(0.80·explicit + 0.20·adjacent, 0, 1)`. No explicit: `score = adjacent × 0.42`. CV/robotics penalty sets `cv_heavy` flag |
| **L3 Fit pillars** | `skill_substance()`, `seniority_fit()`, et al. | 9 weighted pillars; base_fit = Σ weight[k]·pillar[k] |
| **L4 Negative penalties** | `negative_penalty()` | Research-only (no production), entire career in IT-services, CV/robotics focus, job-hopping → multiplicative deductions (up to 0.85 total cap) |
| **L5 Behavioral multiplier** | `behavioral_multiplier()` | Recency of last_active_date (half-life 120 days), recruiter response rate, interview completion rate, open_to_work flag, profile completeness → multiplier ∈ [0.35, 1.0] |

**Final score per candidate:**
```
final = base_fit × role_mult × (1 - neg_penalty) × behavioral_mult
if is_honeypot: final = final × 0.01 − 0.2
```

**Pillar weights (sum = 1.00):**
```
domain_evidence:     0.30   ← calibrated
skill_substance:     0.15   ← calibrated
seniority_fit:       0.12
product_vs_services: 0.10
external_validation: 0.08
eval_frameworks:     0.08   ← calibrated
python_signal:       0.07
location:            0.06
notice:              0.04
```

#### Phase 2 — Population calibration (`population_calibrate()`)
- The three `CALIBRATE_PILLARS` (`domain_evidence`, `skill_substance`, `eval_frameworks`) are replaced with their **percentile rank** across the full 100K population (0 = bottom, 1 = top).
- All other pillars retain JD-relative absolute values.
- Final scores are recomputed with the calibrated pillars.

#### Phase 3 — Isolation Forest (`run_isolation_forest()`)
- Fits an `IsolationForest(n_estimators=120, contamination=0.001, random_state=42)` on 10 numeric features per candidate (yoe, career months, expert-skill-zero-duration count, max skill duration, n_skills, GitHub score, completeness, endorsements, n_current-jobs, duration-ratio).
- Normalises anomaly scores to [0, 1]. Candidates not already caught by L0 with anomaly > 0.85 get an additional multiplicative penalty up to 0.55.
- On this run: **15 candidates** received IF penalties.

#### Phase 4 — Bootstrap confidence (`run_bootstrap()`)
- Generates 500 log-normal weight perturbations around the base `WEIGHTS` vector (seed 42).
- Vectorised matrix multiply: pillar_mat (N×9) @ weight_samples.T (9×500) → (N×500) bootstrap scores.
- `confidence[i]` = fraction of 500 weight configs where candidate i ranks in the top 100.
- Appended to reasoning text: "Rank stable (X% of weight configurations)."

#### Phase 5 — Sort, cap, CSV
- Sort: `(-final_score, candidate_id)` — deterministic tie-break by `candidate_id` ascending.
- Take top 100. Apply monotone non-increasing score cap (ensures spec compliance even after float drift).
- `make_reasoning()` builds the `reasoning` column from the score trace (no hallucination possible — same object).
- `write_csv()` writes exactly 4 columns: `candidate_id, rank, score, reasoning`.

### Total runtime on this machine
**28.1 s** wall-clock (100K candidates, all 5 phases, Apple Silicon, multiprocessing on 10 cores).
This is well within the 5-minute budget.

---

## 3. Local Validation Scorer

Two independent scorers exist:

### validation_harness.py `score` subcommand
```bash
python validation_harness.py score --candidates <file> --labels labels_filled.csv
```
Calls `rank_all()` on the full candidate file, then evaluates against labels.

**Metrics computed:**
| Metric | Weight | Formula |
|---|---|---|
| NDCG@10 | 50% | DCG@10 / IDCG@10 (exponential gain: 2^rel − 1) |
| NDCG@50 | 30% | DCG@50 / IDCG@50 |
| MAP | 15% | AP averaged at each relevant hit (rel_threshold = 3) |
| P@10 | 5% | Fraction of top-10 with tier ≥ 3 |

**Composite formula:**
```
COMPOSITE = 0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10
```

**NDCG@50 included? YES** — it carries 30% of the composite weight.

### score_submission.py
```bash
python3 score_submission.py --submission <csv> --labels labels_filled.csv
```
Scores any pre-built submission CSV (no re-running the ranker). Uses the same composite formula with `REL_THRESHOLD = 4` for MAP and P@10 (stricter than the harness's 3).

**Current scores for `submission_v3_calibrated_v2.csv`:**
```
NDCG@10  : 0.7601   (weight 0.50)
NDCG@50  : 0.6020   (weight 0.30)
MAP      : 0.2096   (weight 0.15, relevant = tier ≥ 4)
P@10     : 1.0000   (weight 0.05)
COMPOSITE: 0.6421
```

---

## 4. Label File — `labels_filled 500.csv`

### Format
Seven columns: `candidate_id, rank, score, tier, tier_csv, tier_xlsx, label_source`

The submission-scoring tools read only `candidate_id` and `tier`.

### Relevance model
**Graded**, not binary. Tiers are integers 0–5:
```
5 = perfect fit      (n = 28)
4 = strong fit       (n = 96)
3 = relevant         (n = 186)
2 = adjacent         (n = 140)
1 = weak match       (n = 50)
0 = honeypot / wrong role  (n = 0 — none in this set)
```
Total: **500 labels**, **0 NaN values**. All 500 candidate IDs are present in `candidates.jsonl`.

### Label provenance
`label_source` column shows dual-labeler reconciliation:
```
both_agree                   : 239
minor_disagreement_bridged   : 240
major_disagreement_bridged   :  21
```
The first 5 labeled IDs (by file order) are the top-5 of the current submission:
`CAND_0064326, CAND_0077337, CAND_0008295, CAND_0010257, CAND_0043637`.

### Additional columns
- `rank` / `score` — the rank and score from `submission_v3_calibrated_v2.csv` at labeling time
- `tier_csv` / `tier_xlsx` — individual labeler votes before bridging

---

## 5. candidates.jsonl — Location and Full Schema

**Path:** `./candidates.jsonl` (100,000 lines, one JSON object per line, no `.gz` compression in this copy)

### Full profile schema (every available field)

```
candidate_id               str        e.g. "CAND_0000001"

profile
  anonymized_name          str        display name (anonymized)
  headline                 str        LinkedIn-style headline
  summary                  str        free-text career summary (can be long)
  location                 str        city / region string
  country                  str        country name
  years_of_experience      float      self-reported total YOE
  current_title            str        most recent job title
  current_company          str        most recent employer
  current_company_size     str        e.g. "201-500", "10001+"
  current_industry         str        e.g. "IT Services", "FinTech"

career_history             list[dict]
  company                  str
  title                    str        job title at this role
  start_date               str        "YYYY-MM-DD"
  end_date                 str|null   null if is_current
  duration_months          int        stated duration in months
  is_current               bool
  industry                 str
  company_size             str
  description              str        free-text role description (key signal for L2)

education                  list[dict]
  institution              str
  degree                   str        e.g. "B.E.", "M.Tech"
  field_of_study           str
  start_year               int
  end_year                 int
  grade                    str        e.g. "8.24 CGPA", "8.5/10"
  tier                     str        e.g. "tier_1", "tier_3" (institution ranking)

skills                     list[dict]
  name                     str        skill label
  proficiency              str        "beginner"|"intermediate"|"advanced"|"expert"
  endorsements             int        LinkedIn-style endorsement count
  duration_months          int        months claimed using this skill

certifications             list[dict] (may be empty)

languages                  list[dict]
  language                 str
  proficiency              str        e.g. "professional", "native"

redrob_signals             dict       platform-side behavioral signals
  profile_completeness_score   float  0–100
  signup_date                  str    "YYYY-MM-DD"
  last_active_date             str    "YYYY-MM-DD"
  open_to_work_flag            bool
  profile_views_received_30d   int
  applications_submitted_30d   int
  recruiter_response_rate      float  0.0–1.0
  avg_response_time_hours      float
  skill_assessment_scores      dict   {skill_name: score_0_to_100}
  connection_count             int
  endorsements_received        int
  notice_period_days           int
  expected_salary_range_inr_lpa  dict  {min: float, max: float}
  preferred_work_mode          str    e.g. "onsite", "hybrid", "remote"
  willing_to_relocate          bool
  github_activity_score        float  0–100 (−1 = not connected)
  search_appearance_30d        int
  saved_by_recruiters_30d      int
  interview_completion_rate    float  0.0–1.0
  offer_acceptance_rate        float  0.0–1.0
  verified_email               bool
  verified_phone               bool
  linkedin_connected           bool
```

**Fields used by rank.py** (subset that actually influences scores):
- `profile`: `years_of_experience`, `current_title`, `headline`, `summary`, `location`, `country`
- `career_history`: `description` (L2 domain signal), `duration_months`, `start_date`, `end_date`, `is_current`, `company`, `industry`
- `skills`: `name`, `proficiency`, `endorsements`, `duration_months`
- `redrob_signals`: `last_active_date`, `recruiter_response_rate`, `interview_completion_rate`, `open_to_work_flag`, `profile_completeness_score`, `github_activity_score`, `willing_to_relocate`, `notice_period_days`, `skill_assessment_scores`

**Fields present but NOT used by rank.py** (available for future use):
`education.tier`, `education.grade`, `certifications`, `languages`, `profile_views_received_30d`, `applications_submitted_30d`, `avg_response_time_hours`, `connection_count`, `expected_salary_range_inr_lpa`, `preferred_work_mode`, `search_appearance_30d`, `saved_by_recruiters_30d`, `offer_acceptance_rate`, `verified_email`, `verified_phone`, `linkedin_connected`

---

## 6. submission_v3_calibrated_v2.csv — Provenance and Format

### How it was produced
Generated by running `rank.py` with `candidates.jsonl` as input, all 5 phases active (population calibration + Isolation Forest + bootstrap confidence). The "v3_calibrated_v2" name indicates this was the second revision of the v3 calibrated submission.

Reproduce command:
```bash
python rank.py --candidates candidates.jsonl --out submission_v3_calibrated_v2.csv
```

### Column set
Exactly 4 columns as required by spec:
```
candidate_id   rank   score   reasoning
```

### Row count and score properties
- **100 rows** (ranks 1–100, each rank appears exactly once)
- Score range: **0.726068 – 0.880094**
- Score is **monotone non-increasing** ✓ (verified programmatically)
- Tie-break: `candidate_id` ascending (deterministic) ✓

### Top-5 candidates
```
CAND_0064326  rank=1  score=0.880094
CAND_0077337  rank=2  score=0.867864
CAND_0008295  rank=3  score=0.853768
CAND_0010257  rank=4  score=0.837848
CAND_0043637  rank=5  score=0.819822
```

---

## 7. Network / LLM Usage Grep

Search targets: all Python files in the ranking path (`rank.py`, `validation_harness.py`, `score_submission.py`, `train_ltr.py`, `tune.py`).

Search terms: `requests`, `http.`, `openai`, `anthropic`, `cohere`, `google.generativeai`, `genai`, `langchain`, `httpx`, `urllib`

```
RESULT: NO MATCHES
```

The ranking path is entirely offline. The only third-party libraries imported are:
- `numpy` — numerical computation
- `scikit-learn` — `IsolationForest`, `StandardScaler` (Phase 3; falls back gracefully if absent)
- `lightgbm` — used only in `train_ltr.py` (optional LambdaMART step)
- `pandas` — used only in `score_submission.py` (scoring utility)

No network calls, no LLM API calls, no embedding model weights anywhere in the ranking path.
