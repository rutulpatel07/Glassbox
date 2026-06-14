"""
ml_engine.py
============
Pure mathematical ML hiring engine — zero LLM, zero API calls.

Algorithms Used
───────────────
1.  TF-IDF Vectorization          Converts skill lists into weighted term vectors
2.  Cosine Similarity             Geometric angle between candidate & job skill vectors
3.  Gaussian Bell Curve           Scores experience vs ideal range (peaks at midpoint)
4.  Logistic / Sigmoid            GPA normalization and availability scoring
5.  Tiered Scoring Matrix         Degree, institution, company prestige lookup tables
6.  Exponential Recency Weighting Latest jobs count more in career trajectory score
7.  MiniBatch K-Means Clustering  Unsupervised segmentation → Excellent/Strong/Average/Weak
8.  Z-Score Anomaly Detection     Flags statistical outliers in salary/experience fields
9.  Rankdata Percentile Ranking   O(n log n) percentile vs O(n²) naive loop
10. Weighted MCDM                 Multi-Criteria Decision Making via dot product
"""

import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata, zscore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")


# ── Lookup tables (used as in-memory scoring matrices) ────────────────────────

DEGREE_SCORE = {
    "Ph.D":   1.00, "M.Tech": 0.90, "M.Sc":   0.86,
    "MBA":    0.85, "MCA":    0.80, "M.Com":  0.74,
    "B.Tech": 0.75, "B.E":    0.75, "B.Sc":   0.65,
    "BCA":    0.60, "B.Com":  0.55, "Diploma":0.45,
}

INSTITUTION_TIER_SCORE = {"tier1": 1.00, "tier2": 0.75, "tier3": 0.45}
COMPANY_TIER_SCORE     = {"tier1": 1.00, "tier2": 0.68, "tier3": 0.38}

# Skill synonym groups for soft matching
SKILL_SYNONYMS = {
    "machine_learning": ["ml","machine learning","deep learning","ai","artificial intelligence"],
    "python":           ["python","python3","python2"],
    "javascript":       ["javascript","js","typescript","ts"],
    "cloud":            ["aws","gcp","azure","cloud","cloud computing"],
    "database":         ["sql","postgresql","mysql","database","nosql"],
    "containers":       ["docker","kubernetes","k8s","container"],
    "data_science":     ["data science","data analysis","analytics","statistics"],
    "nlp":              ["nlp","natural language processing","text mining"],
}


class HiringMLEngine:
    """
    End-to-end ML candidate ranking pipeline.

    Usage
    -----
    engine = HiringMLEngine(config)
    df     = engine.load_candidates("candidates.json")
    job    = engine.load_job("sample_job.json")
    top, full = engine.run(df, job, top_n=100)
    engine.export_results(top, "results.json")
    engine.print_summary(top)
    """

    def __init__(self, config: dict):
        self.config = config
        self.scaler  = MinMaxScaler()
        self._tfidf  = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=8000,
            lowercase=True,
            sublinear_tf=True,           # log(1+tf) dampens dominating terms
        )

    # ──────────────────────────────────────────────────────────────────────────
    # DATA LOADING
    # ──────────────────────────────────────────────────────────────────────────

    def load_candidates(self, path: str) -> pd.DataFrame:
        print(f"Loading candidates …  {path}")
        with open(path, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        print(f"  ✓ {len(df):,} candidates loaded  |  columns: {list(df.columns)}")
        return df

    def load_job(self, path: str) -> dict:
        with open(path, "r") as f:
            job = json.load(f)
        print(f"  ✓ Job: [{job['title']}]")
        return job

    # ──────────────────────────────────────────────────────────────────────────
    # 1. SKILL SCORE  ── TF-IDF + Cosine Similarity
    # ──────────────────────────────────────────────────────────────────────────

    def _skills_to_doc(self, skills) -> str:
        """
        Converts a skill list to a TF-IDF document.
        Also injects synonym tokens so 'ML' ≈ 'Machine Learning'.
        """
        if not isinstance(skills, list):
            return " "
        tokens = []
        for sk in skills:
            tok = sk.lower().replace(" ", "_").replace(".", "").replace("/", "_")
            tokens.append(tok)
            # Synonym expansion
            for canonical, synonyms in SKILL_SYNONYMS.items():
                if sk.lower() in synonyms:
                    tokens.append(canonical)
        return " ".join(tokens) if tokens else " "

    def score_skills(self, df: pd.DataFrame, job: dict) -> np.ndarray:
        """
        Algorithm
        ─────────
        • Build a TF-IDF corpus from every candidate's skill list + job skills.
        • Compute cosine similarity of each candidate vector vs the job vector.
        • Add hard-match bonuses:
            – required_bonus  : fraction of required skills present  (+0.35 weight)
            – preferred_bonus : fraction of preferred skills present  (+0.10 weight)
        • Normalize result to [0, 1].
        """
        req_skills  = [s.lower() for s in job.get("required_skills",  [])]
        pref_skills = [s.lower() for s in job.get("preferred_skills", [])]
        all_job_skills = req_skills + pref_skills

        candidate_docs = df["skills"].apply(self._skills_to_doc).tolist()
        job_doc = " ".join(
            [s.replace(" ", "_").replace(".", "") for s in all_job_skills]
        )

        all_docs = candidate_docs + [job_doc]
        all_docs = [d if d.strip() else " " for d in all_docs]

        try:
            tfidf_matrix  = self._tfidf.fit_transform(all_docs)
        except Exception:
            return np.zeros(len(df))

        job_vec        = tfidf_matrix[-1]         # last row is the job
        candidate_vecs = tfidf_matrix[:-1]

        # Base cosine similarity  (shape: N,)
        cos_scores = cosine_similarity(candidate_vecs, job_vec).ravel()

        # Hard-match bonuses (vectorized via numpy)
        req_bonus  = np.zeros(len(df))
        pref_bonus = np.zeros(len(df))

        for idx, skills in enumerate(df["skills"]):
            if not isinstance(skills, list):
                continue
            cand_lower = [s.lower() for s in skills]
            # Count hits (partial match — checks if req skill appears inside any cand skill)
            rh = sum(1 for rs in req_skills  if any(rs in cs for cs in cand_lower))
            ph = sum(1 for ps in pref_skills if any(ps in cs for cs in cand_lower))
            req_bonus[idx]  = (rh / max(len(req_skills),  1)) * 0.35
            pref_bonus[idx] = (ph / max(len(pref_skills), 1)) * 0.10

        raw = 0.55 * cos_scores + req_bonus + pref_bonus
        mx  = raw.max()
        return (raw / mx) if mx > 0 else raw

    # ──────────────────────────────────────────────────────────────────────────
    # 2. EXPERIENCE SCORE  ── Gaussian Bell Curve
    # ──────────────────────────────────────────────────────────────────────────

    def score_experience(self, df: pd.DataFrame, job: dict) -> np.ndarray:
        """
        Formula: score = exp( -0.5 * ((years - μ) / σ)² )
          μ = ideal experience = midpoint of [min, max]
          σ = (max - min) / 3   (3-sigma rule covers 99.7% of acceptable range)

        Below-minimum candidates are penalised quadratically.
        """
        lo  = job.get("min_experience", 0)
        hi  = job.get("max_experience", 20)
        mu  = (lo + hi) / 2.0
        sig = max((hi - lo) / 3.0, 1.0)

        yrs = df["experience_years"].fillna(0).values.astype(float)

        # Gaussian
        scores = np.exp(-0.5 * ((yrs - mu) / sig) ** 2)

        # Extra penalty for hard under-qualification
        mask = yrs < lo
        scores[mask] *= (np.maximum(yrs[mask], 0) / max(lo, 1)) ** 2

        return scores

    # ──────────────────────────────────────────────────────────────────────────
    # 3. EDUCATION SCORE  ── Tiered Lookup + Sigmoid GPA
    # ──────────────────────────────────────────────────────────────────────────

    def score_education(self, df: pd.DataFrame, job: dict) -> np.ndarray:
        """
        Four sub-scores combined with fixed weights:
          35% degree level match
          25% institution tier
          20% field relevance
          20% GPA (logistic sigmoid around 7.0/10)
        """
        req_degree  = job.get("education_requirement", "B.Tech")
        req_fields  = [f.lower() for f in job.get("preferred_fields", [])]
        req_d_score = DEGREE_SCORE.get(req_degree, 0.60)

        edu = df["education"]

        degrees   = edu.apply(lambda x: x.get("degree",           "")       if isinstance(x, dict) else "")
        inst_tier = edu.apply(lambda x: x.get("institution_tier", "tier3")  if isinstance(x, dict) else "tier3")
        fields    = edu.apply(lambda x: x.get("field",            "").lower() if isinstance(x, dict) else "")
        gpas      = edu.apply(lambda x: x.get("gpa",              7.0)      if isinstance(x, dict) else 7.0)

        # Degree match (capped at 1.0 — overqualification is fine)
        d_scores  = degrees.map(lambda d: DEGREE_SCORE.get(d, 0.50)).values
        d_match   = np.minimum(d_scores / req_d_score, 1.0)

        # Institution tier
        t_scores  = inst_tier.map(lambda t: INSTITUTION_TIER_SCORE.get(t, 0.45)).values

        # Field relevance
        if req_fields:
            f_scores = fields.apply(
                lambda f: 1.0 if any(rf in f for rf in req_fields) else 0.55
            ).values
        else:
            f_scores = np.full(len(df), 0.75)

        # GPA  → sigmoid: score = 1 / (1 + exp(-k*(gpa - midpoint)))
        #   k=0.9 gives steep rise around 7.0; max ≈ 1, min ≈ 0
        gpa_arr   = np.clip(gpas.values.astype(float), 0.0, 10.0)
        g_scores  = 1.0 / (1.0 + np.exp(-0.9 * (gpa_arr - 7.0)))

        return 0.35 * d_match + 0.25 * t_scores + 0.20 * f_scores + 0.20 * g_scores

    # ──────────────────────────────────────────────────────────────────────────
    # 4. SALARY FIT SCORE  ── Gaussian around midpoint
    # ──────────────────────────────────────────────────────────────────────────

    def score_salary_fit(self, df: pd.DataFrame, job: dict) -> np.ndarray:
        """
        Gaussian centred at the midpoint of the salary band.
        σ = (max - min) / 4  →  candidates outside ±2σ score < 0.14

        Hard rules:
          > max * 1.25  → score = 0.0  (too expensive)
          < min * 0.55  → score *= 0.8  (may leave for better pay)
        """
        lo, hi = job.get("salary_range", [0, 10_000_000])
        mid    = (lo + hi) / 2.0
        sig    = max((hi - lo) / 4.0, 50_000)

        sal = df["salary_expectation"].fillna(mid).values.astype(float)
        scores = np.exp(-0.5 * ((sal - mid) / sig) ** 2)

        scores[sal > hi * 1.25] = 0.0
        scores[sal < lo * 0.55] *= 0.80
        return scores

    # ──────────────────────────────────────────────────────────────────────────
    # 5. LOCATION SCORE
    # ──────────────────────────────────────────────────────────────────────────

    def score_location(self, df: pd.DataFrame, job: dict) -> np.ndarray:
        """
        • Exact city match       → 1.00
        • Willing to relocate    → 0.75
        • Different city, no relo→ 0.25
        """
        job_loc   = job.get("location", "").lower()
        locs      = df["location"].fillna("").str.lower().values
        relocate  = df["willing_to_relocate"].fillna(False).values.astype(bool)

        exact     = np.array([job_loc in l or l in job_loc for l in locs])
        scores    = np.where(exact, 1.00,
                   np.where(relocate, 0.75, 0.25))
        return scores.astype(float)

    # ──────────────────────────────────────────────────────────────────────────
    # 6. CAREER TRAJECTORY SCORE  ── Exponential Recency Weighting
    # ──────────────────────────────────────────────────────────────────────────

    def _trajectory_single(self, history) -> float:
        """
        Sub-scores for one candidate:
          • Company tier quality (exp-weighted so recent jobs matter more)
          • Job-hopping penalty: any stint < 18 months reduces score
        """
        if not isinstance(history, list) or len(history) == 0:
            return 0.50

        tiers = [
            COMPANY_TIER_SCORE.get(
                w.get("company_tier", "tier3") if isinstance(w, dict) else "tier3",
                0.38
            )
            for w in history
        ]
        n = len(tiers)
        # Exponential recency weights: last job is most important
        exp_w = np.exp(np.linspace(0, 1, n))
        exp_w /= exp_w.sum()
        weighted_tier = float(np.dot(exp_w, tiers))

        # Hopping penalty
        short = sum(
            1 for w in history
            if isinstance(w, dict) and w.get("years", 2) < 1.5
        )
        hop_penalty = 1.0 - (short / n) * 0.50

        return 0.55 * weighted_tier + 0.45 * hop_penalty

    def score_career_trajectory(self, df: pd.DataFrame) -> np.ndarray:
        return df["work_history"].apply(self._trajectory_single).values

    # ──────────────────────────────────────────────────────────────────────────
    # 7. CERTIFICATION SCORE
    # ──────────────────────────────────────────────────────────────────────────

    def score_certifications(self, df: pd.DataFrame, job: dict) -> np.ndarray:
        """
        • Base  = min(num_certs * 0.12, 0.45)  — rewards breadth
        • Bonus = fraction of job-relevant certs present
        • Combined 40/60 weighted sum
        """
        rel_certs = [c.lower() for c in job.get("relevant_certifications", [])]

        def _score(certs) -> float:
            if not isinstance(certs, list) or len(certs) == 0:
                return 0.25          # neutral, not penalised
            base = min(len(certs) * 0.12, 0.45)
            if not rel_certs:
                return min(base + 0.30, 1.0)
            cl = [c.lower() for c in certs]
            hits = sum(1 for rc in rel_certs if any(rc in c for c in cl))
            relevance = hits / len(rel_certs)
            return float(np.clip(0.40 * base + 0.60 * relevance, 0, 1))

        return df["certifications"].apply(_score).values

    # ──────────────────────────────────────────────────────────────────────────
    # 8. AVAILABILITY SCORE  ── Sigmoid on notice period
    # ──────────────────────────────────────────────────────────────────────────

    def score_availability(self, df: pd.DataFrame, job: dict) -> np.ndarray:
        """
        score = 1 / (1 + exp( k * (notice - max_notice) ))
        Scores > 0.5 when notice < max_notice; drops sharply beyond.
        """
        max_notice = float(job.get("max_notice_period_days", 60))
        notice     = df["notice_period_days"].fillna(30).values.astype(float)
        k = 0.10
        return 1.0 / (1.0 + np.exp(k * (notice - max_notice)))

    # ──────────────────────────────────────────────────────────────────────────
    # COMPOSITE SCORING  ── Weighted MCDM
    # ──────────────────────────────────────────────────────────────────────────

    def composite_score(self, score_matrix: np.ndarray, weights: list) -> np.ndarray:
        """
        Weighted sum:  score_i = Σ_k  weight_k × criterion_k_i

        weights are normalised to sum=1 internally, so config values
        don't need to be exact.
        """
        w = np.array(weights, dtype=float)
        w = w / w.sum()
        return score_matrix @ w          # shape (N,)

    # ──────────────────────────────────────────────────────────────────────────
    # K-MEANS CLUSTERING  ── Candidate Tier Segmentation
    # ──────────────────────────────────────────────────────────────────────────

    def cluster_candidates(self, score_matrix: np.ndarray, n_clusters: int = 4) -> list:
        """
        MiniBatchKMeans on the full score matrix (faster for 1L records).
        Assigns tier names by ordering cluster centroids from best to worst.
        """
        km = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=42,
            batch_size=5000, n_init=10
        )
        labels = km.fit_predict(score_matrix)

        # Rank clusters by mean centroid value → best cluster = "Excellent"
        centroid_means = km.cluster_centers_.mean(axis=1)
        rank_order     = np.argsort(centroid_means)[::-1]   # descending

        tier_names = ["Excellent", "Strong", "Average", "Weak"]
        label_map  = {orig: tier_names[i] for i, orig in enumerate(rank_order)}

        return [label_map.get(l, "Unknown") for l in labels]

    # ──────────────────────────────────────────────────────────────────────────
    # Z-SCORE ANOMALY DETECTION
    # ──────────────────────────────────────────────────────────────────────────

    def detect_anomalies(self, df: pd.DataFrame) -> list:
        """
        Uses z-scores to flag statistical anomalies:
          • |z_experience| > 3.0        → extreme_experience
          • Low exp but very high salary → salary_experience_mismatch
          • Very high exp but few skills → low_skills_high_experience
        """
        exp_col = df["experience_years"].fillna(0).values.astype(float)
        sal_col = df["salary_expectation"].fillna(0).values.astype(float)

        z_exp = np.abs(zscore(exp_col))
        z_sal = np.abs(zscore(sal_col))

        flags = []
        for i in range(len(df)):
            f = []
            if z_exp[i] > 3.0:
                f.append("extreme_experience")
            if exp_col[i] < 2 and sal_col[i] > 2_500_000:
                f.append("salary_experience_mismatch")
            skills = df.iloc[i]["skills"]
            if exp_col[i] > 10 and isinstance(skills, list) and len(skills) < 3:
                f.append("low_skills_high_experience")
            if z_sal[i] > 3.5:
                f.append("extreme_salary")
            flags.append(f)
        return flags

    # ──────────────────────────────────────────────────────────────────────────
    # FEATURE IMPORTANCE  ── Pearson Correlation
    # ──────────────────────────────────────────────────────────────────────────

    def feature_importance(self, score_matrix: np.ndarray,
                            composite: np.ndarray,
                            criteria_names: list) -> dict:
        """
        Pearson correlation of each criterion with the composite score.
        Tells you which factors drive the ranking most.
        """
        importance = {}
        for idx, name in enumerate(criteria_names):
            col = score_matrix[:, idx]
            if col.std() > 0:
                r = float(np.corrcoef(col, composite)[0, 1])
            else:
                r = 0.0
            importance[name] = round(abs(r), 4)
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    # ──────────────────────────────────────────────────────────────────────────
    # FULL PIPELINE
    # ──────────────────────────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame, job: dict,
            top_n: int = 100,
            filter_domain: str = None):
        """
        Executes the complete ML pipeline in sequence:
          1  Optional domain filter
          2  Compute 8 component scores
          3  Assemble score matrix (N × 8)
          4  Weighted composite score
          5  Percentile ranking  (O(n log n) via rankdata)
          6  MiniBatch K-Means tier clustering
          7  Z-Score anomaly detection
          8  Feature importance (Pearson r)
          9  Sort + return top_n results
        """
        print(f"\n{'━'*65}")
        print(f"  ML HIRING ENGINE  ─  {job['title']}")
        print(f"{'━'*65}")

        # ── Optional domain filter ──
        if filter_domain:
            df = df[df["domain"] == filter_domain].reset_index(drop=True)
            print(f"  Domain filter [{filter_domain}]: {len(df):,} candidates")

        N = len(df)
        print(f"  Candidates to rank: {N:,}")
        print()

        W = self.config.get("weights", {})

        # ── 8 Scoring steps ──
        print("  [1/8] Skill Match       (TF-IDF + Cosine Similarity) …")
        s_skill   = self.score_skills(df, job)

        print("  [2/8] Experience        (Gaussian Bell Curve) …")
        s_exp     = self.score_experience(df, job)

        print("  [3/8] Education         (Tiered + Sigmoid GPA) …")
        s_edu     = self.score_education(df, job)

        print("  [4/8] Salary Fit        (Gaussian around midpoint) …")
        s_sal     = self.score_salary_fit(df, job)

        print("  [5/8] Location          (Exact/Relocation) …")
        s_loc     = self.score_location(df, job)

        print("  [6/8] Career Trajectory (Exp-Weighted Company Tiers) …")
        s_traj    = self.score_career_trajectory(df)

        print("  [7/8] Certifications    (Relevance + Breadth) …")
        s_cert    = self.score_certifications(df, job)

        print("  [8/8] Availability      (Sigmoid Notice Period) …")
        s_avail   = self.score_availability(df, job)

        # ── Score matrix (N × 8) ──
        CRITERIA = [
            "skills","experience","education","salary_fit",
            "location","career_trajectory","certifications","availability"
        ]
        score_matrix = np.column_stack(
            [s_skill, s_exp, s_edu, s_sal, s_loc, s_traj, s_cert, s_avail]
        )

        # ── Weights ──
        weights = [
            W.get("skills",            0.30),
            W.get("experience",        0.25),
            W.get("education",         0.15),
            W.get("salary_fit",        0.10),
            W.get("location",          0.08),
            W.get("career_trajectory", 0.07),
            W.get("certifications",    0.03),
            W.get("availability",      0.02),
        ]

        # ── Composite score ──
        print("\n  ► Computing composite (MCDM weighted sum) …")
        composite = self.composite_score(score_matrix, weights)

        # ── Percentile rank via rankdata  O(n log n) ──
        print("  ► Percentile ranking (rankdata) …")
        percentiles = (rankdata(composite) / N) * 100

        # ── K-Means clustering ──
        print("  ► K-Means tier clustering (MiniBatch, k=4) …")
        tier_labels = self.cluster_candidates(score_matrix)

        # ── Anomaly detection ──
        print("  ► Z-Score anomaly detection …")
        anomalies = self.detect_anomalies(df)

        # ── Feature importance ──
        importance = self.feature_importance(score_matrix, composite, CRITERIA)

        # ── Assemble result dataframe ──
        result_df = df.copy()
        result_df["score_skill"]         = np.round(s_skill,  4)
        result_df["score_experience"]    = np.round(s_exp,    4)
        result_df["score_education"]     = np.round(s_edu,    4)
        result_df["score_salary_fit"]    = np.round(s_sal,    4)
        result_df["score_location"]      = np.round(s_loc,    4)
        result_df["score_trajectory"]    = np.round(s_traj,   4)
        result_df["score_certifications"]= np.round(s_cert,   4)
        result_df["score_availability"]  = np.round(s_avail,  4)
        result_df["composite_score"]     = np.round(composite,  4)
        result_df["percentile_rank"]     = np.round(percentiles, 2)
        result_df["candidate_tier"]      = tier_labels
        result_df["anomaly_flags"]       = anomalies

        result_df = result_df.sort_values(
            "composite_score", ascending=False
        ).reset_index(drop=True)
        result_df["rank"] = result_df.index + 1

        # ── Stats ──
        print(f"\n  {'─'*50}")
        print(f"  Score range : {composite.min():.4f}  →  {composite.max():.4f}")
        print(f"  Mean score  : {composite.mean():.4f}    Std: {composite.std():.4f}")
        print(f"  {'─'*50}")
        print("  Feature Importance (Pearson r vs composite):")
        for crit, r_val in importance.items():
            bar = "▓" * int(r_val * 30)
            print(f"    {crit:<22}  {r_val:.4f}  {bar}")

        top = result_df.head(top_n).copy()
        return top, result_df, importance

    # ──────────────────────────────────────────────────────────────────────────
    # EXPORT
    # ──────────────────────────────────────────────────────────────────────────

    def export_results(self, top: pd.DataFrame, path: str = "results.json") -> None:
        records = []
        for _, row in top.iterrows():
            records.append({
                "rank":            int(row["rank"]),
                "id":              row["id"],
                "name":            row["name"],
                "composite_score": float(row["composite_score"]),
                "percentile_rank": float(row["percentile_rank"]),
                "candidate_tier":  row["candidate_tier"],
                "anomaly_flags":   row["anomaly_flags"],
                "scores": {
                    "skills":            float(row["score_skill"]),
                    "experience":        float(row["score_experience"]),
                    "education":         float(row["score_education"]),
                    "salary_fit":        float(row["score_salary_fit"]),
                    "location":          float(row["score_location"]),
                    "career_trajectory": float(row["score_trajectory"]),
                    "certifications":    float(row["score_certifications"]),
                    "availability":      float(row["score_availability"]),
                },
                "profile": {
                    "skills":           row["skills"],
                    "experience_years": int(row["experience_years"]),
                    "location":         row["location"],
                    "salary_expectation": int(row["salary_expectation"]),
                    "notice_period_days": int(row["notice_period_days"]),
                    "certifications":   row["certifications"],
                    "education":        row["education"],
                    "work_history":     row["work_history"],
                }
            })

        with open(path, "w") as f:
            json.dump(records, f, indent=2, default=str)
        print(f"\n  ✓ Results written → {path}  ({len(records)} candidates)")

    # ──────────────────────────────────────────────────────────────────────────
    # PRETTY PRINT SUMMARY
    # ──────────────────────────────────────────────────────────────────────────

    def print_summary(self, top: pd.DataFrame, show: int = 25) -> None:
        print(f"\n{'━'*100}")
        print(f"{'RNK':<5} {'NAME':<26} {'SCORE':<7} {'PCTILE':<8} {'TIER':<12} "
              f"{'EXP':<5} {'LOC':<14} {'TOP SKILLS'}")
        print(f"{'━'*100}")

        for _, r in top.head(show).iterrows():
            skills_str = ", ".join(r["skills"][:4]) if isinstance(r["skills"], list) else ""
            if isinstance(r["skills"], list) and len(r["skills"]) > 4:
                skills_str += " …"
            loc = str(r.get("location", ""))[:13]
            print(
                f"{int(r['rank']):<5} {r['name']:<26} "
                f"{r['composite_score']:<7.4f} {r['percentile_rank']:<8.1f} "
                f"{r['candidate_tier']:<12} {r['experience_years']:<5} "
                f"{loc:<14} {skills_str}"
            )

        print(f"{'━'*100}")

        # Tier distribution bar chart
        tier_dist = top["candidate_tier"].value_counts()
        print(f"\n  Tier Distribution  (top {len(top)}):")
        for tier in ["Excellent","Strong","Average","Weak"]:
            cnt = tier_dist.get(tier, 0)
            bar = "█" * max(int(cnt / max(tier_dist.values) * 40), 0)
            print(f"    {tier:<12}  {cnt:>5}  {bar}")

        # Anomaly count
        flagged = top["anomaly_flags"].apply(lambda x: len(x) > 0).sum()
        if flagged:
            print(f"\n  ⚠  {flagged} candidate(s) in top results have anomaly flags")
        print()
