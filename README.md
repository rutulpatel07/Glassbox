# ML Hiring Engine
### Rank 1 Lakh Candidates — Pure Math, Zero LLM

Processes 100,000 candidates in **~15 seconds** using only classical ML algorithms and linear algebra. No API calls, no neural networks, no internet connection required.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Step 1 — Generate 1 lakh synthetic candidates
python main.py --generate --count 100000

# Step 2 — Rank them for a job
python main.py --candidates candidates.json --job sample_job.json --top 100

# Step 3 — View results
cat results.json
```

---

## Project Structure

```
hiring_ml_engine/
├── generate_candidates.py   # Synthetic dataset generator (1L candidates)
├── ml_engine.py             # Core ML pipeline (all algorithms live here)
├── main.py                  # CLI entry point
├── config.json              # Tunable weights (no code change needed)
├── sample_job.json          # Example: Senior ML Engineer
├── jobs/
│   └── product_manager.json # Example: Senior Product Manager
├── requirements.txt
└── README.md
```

---

## ML Algorithms Used

| # | Algorithm | Where Used | Math |
|---|-----------|------------|------|
| 1 | **TF-IDF Vectorization** | Skill documents | `tf(t,d) × log(N/df(t))` |
| 2 | **Cosine Similarity** | Skill match score | `cos(θ) = A·B / (‖A‖ ‖B‖)` |
| 3 | **Gaussian Bell Curve** | Experience scoring | `exp(-½((x-μ)/σ)²)` |
| 4 | **Logistic / Sigmoid** | GPA + availability | `1 / (1 + e^(-k(x-c)))` |
| 5 | **Tiered Scoring Matrix** | Degree / institution / company | Lookup table → float |
| 6 | **Exponential Recency Weighting** | Career trajectory | `w_i = e^(i/n) / Σe^(i/n)` |
| 7 | **MiniBatch K-Means** | Candidate tier clustering | `min Σ ‖x - μ_k‖²` |
| 8 | **Z-Score Anomaly Detection** | Suspicious profiles | `z = (x - μ) / σ` |
| 9 | **Rankdata Percentile** | Percentile ranking | `rank(x) / N × 100` |
| 10 | **Weighted MCDM** | Final composite score | `score = W · S_vec` |
| 11 | **Pearson Correlation** | Feature importance | `r = cov(X,Y) / (σ_X σ_Y)` |

---

## Scoring Breakdown

### 1. Skill Score (weight: 30%)
Builds a TF-IDF corpus from all candidate skill lists. The job's required + preferred skills form the query vector. Cosine similarity measures angle between vectors — 0 = orthogonal (no overlap), 1 = identical.

Hard-match bonuses layer on top:
- +0.35 × (fraction of required skills present)
- +0.10 × (fraction of preferred skills present)

### 2. Experience Score (weight: 25%)
Gaussian bell curve centred at the **midpoint** of `[min_experience, max_experience]`. Standard deviation = `(max - min) / 3` (3-sigma rule). Candidates below the minimum are penalised quadratically.

```
ideal = (min + max) / 2
score = exp(-0.5 × ((years - ideal) / sigma)²)
```

### 3. Education Score (weight: 15%)
Four sub-components weighted internally:
- **35%** — Degree level match (PhD=1.0 → Diploma=0.45)
- **25%** — Institution tier (IIT/IIM=1.0 → State college=0.45)
- **20%** — Field relevance to job
- **20%** — GPA via sigmoid (steep rise around 7.0/10)

### 4. Salary Fit Score (weight: 10%)
Gaussian around salary band midpoint (σ = range/4). Hard rules:
- Candidate asks > 1.25× max salary → score = 0 (unaffordable)
- Candidate asks < 0.55× min salary → score × 0.8 (flight risk)

### 5. Location Score (weight: 8%)
- Exact city match → 1.00
- Willing to relocate → 0.75
- Different city, no relocation → 0.25

### 6. Career Trajectory Score (weight: 7%)
Two factors combined:
- **Company tier quality** — averaged with exponential recency weights (most recent job matters most)
- **Job hopping penalty** — each stint under 18 months reduces score by 50%/n

### 7. Certification Score (weight: 3%)
Base score for breadth (more certs = higher base, capped). Bonus for hitting the job's `relevant_certifications` list.

### 8. Availability Score (weight: 2%)
Sigmoid decay on notice period vs job's `max_notice_period_days`. Drops sharply beyond the threshold.

---

## Composite Score Formula

```
composite = W · [s_skill, s_exp, s_edu, s_salary, s_loc, s_traj, s_cert, s_avail]ᵀ

W is normalised: W / sum(W)   (so config values don't need to sum exactly to 1)
```

---

## K-Means Tier Clustering

After scoring, `MiniBatchKMeans(k=4)` clusters candidates into tiers based on their full 8-dimensional score vector — not just the composite. Cluster centroids are ranked by mean value to assign:

```
Excellent  →  top cluster centroid
Strong     →  second cluster
Average    →  third cluster
Weak       →  bottom cluster
```

---

## Anomaly Detection

Z-score flags are added to each candidate but do **not** remove them from ranking (recruiter decision). Three checks:

| Flag | Condition |
|------|-----------|
| `extreme_experience` | \|z(years)\| > 3.0 |
| `salary_experience_mismatch` | < 2 years exp but salary > ₹25 LPA |
| `low_skills_high_experience` | > 10 years exp but < 3 skills listed |

---

## Output Files

**`results.json`** — Top N candidates with full breakdown:
```json
{
  "rank": 1,
  "composite_score": 0.8245,
  "percentile_rank": 99.99,
  "candidate_tier": "Excellent",
  "scores": { "skills": 0.92, "experience": 0.98, ... },
  "profile": { "skills": [...], "experience_years": 8, ... },
  "anomaly_flags": []
}
```

**`results_stats.json`** — Run statistics:
```json
{
  "total_candidates": 100000,
  "processing_time_sec": 15.6,
  "candidates_per_sec": 6400,
  "feature_importance": { "experience": 0.81, "skills": 0.49, ... }
}
```

---

## CLI Options

```
--candidates  Path to candidates JSON        (default: candidates.json)
--job         Path to job requirements JSON  (default: sample_job.json)
--config      Path to weights config JSON    (default: config.json)
--top         How many results to return     (default: 100)
--output      Output file path               (default: results.json)
--domain      Filter by domain               (tech/management/design/finance/marketing)
--show        Rows to print in summary table (default: 20)
--generate    Generate synthetic candidates first
--count       Number of candidates to gen    (default: 100000)
```

---

## Using Your Own Candidate Data

Your `candidates.json` must be a JSON array. Each object should have:

```json
{
  "id": "C0000001",
  "name": "Rahul Sharma",
  "location": "Bangalore",
  "willing_to_relocate": true,
  "skills": ["Python", "Machine Learning", "TensorFlow"],
  "experience_years": 6,
  "work_history": [
    { "company": "Google", "company_tier": "tier1", "role": "ML Engineer", "years": 3 }
  ],
  "education": {
    "degree": "B.Tech",
    "field": "Computer Science",
    "institution": "IIT Bombay",
    "institution_tier": "tier1",
    "gpa": 9.1
  },
  "certifications": ["AWS Solutions Architect Associate"],
  "salary_expectation": 2500000,
  "notice_period_days": 30,
  "domain": "tech"
}
```

Missing fields are handled gracefully with neutral fallback values.

---

## Tuning Weights

Edit `config.json` — no code change needed:

```json
{
  "weights": {
    "skills":            0.40,   ← increase if skill match is most critical
    "experience":        0.20,
    "education":         0.10,
    "salary_fit":        0.10,
    "location":          0.05,
    "career_trajectory": 0.08,
    "certifications":    0.05,
    "availability":      0.02
  }
}
```

---

## Performance

| Dataset size | Processing time | Throughput |
|-------------|----------------|-----------|
| 5,000       | ~1 second      | ~5,000/sec |
| 100,000     | ~16 seconds    | ~6,400/sec |

Tested on a standard CPU. All operations are vectorized via NumPy/Pandas/scikit-learn.

---

## Dependencies

```
numpy       — vectorised math
pandas      — dataframe operations
scikit-learn — TF-IDF, Cosine Similarity, MiniBatchKMeans
scipy       — rankdata, zscore
```

No internet, no GPU, no paid APIs required.
