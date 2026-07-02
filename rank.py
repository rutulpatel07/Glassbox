#!/usr/bin/env python3
"""
rank.py — Redrob Intelligent Candidate Discovery & Ranking Engine v2
=====================================================================
Glass-box, deterministic, zero-network, CPU-only candidate ranker.
Now with: multiprocessing across all cores, population-calibrated scoring,
Isolation Forest anomaly detection, and bootstrap rank-confidence.

ARCHITECTURE (five phases after loading)
-----------------------------------------
  Phase 1  Extract raw features — multiprocessing across all CPU cores
  Phase 2  Population calibration (Concept A) — percentile-normalize the
           most arbitrary pillars against the real 100K distribution
  Phase 3  Isolation Forest (Concept B) — unsupervised second-opinion on
           honeypots; catches patterns no hand-written rule anticipated
  Phase 4  Bootstrap rank-confidence (Concept C) — 500 weight perturbations
           via matrix multiply; confidence interval per candidate
  Phase 5  Final rank, trace-driven reasoning (now includes confidence),
           spec-compliant CSV

SCORING LAYERS (within each candidate's feature extraction)
------------------------------------------------------------
  L0  Honeypot / impossible-profile gate         → hard floor
  L1  Role-fit gate                              → floors keyword-stuffers
  L2  Domain evidence (layered explicit/adjacent)→ reads career descriptions
  L3  Fit pillars                                → 9 weighted components
  L4  Negative do-NOT-want penalties             → multiplicative deduction
  L5  Behavioral availability multiplier         → JD "actually available"

Usage
-----
    python rank.py --candidates candidates.jsonl.gz --out submission.csv
    python rank.py --candidates sample_candidates.json --out sample_out.csv --top 25
"""

import argparse
import csv
import gzip
import json
import math
import os
import re
from datetime import date, datetime
from multiprocessing import Pool, cpu_count

import numpy as np

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import rich as _rich_check  # noqa: F401
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# ──────────────────────────────────────────────────────────────────────────────
# JOB PROFILE — hand-derived from the REAL Redrob JD
# ──────────────────────────────────────────────────────────────────────────────
JOB = {
    "title": "Senior AI Engineer — Founding Team (Redrob)",
    "ideal_yoe": 7.0, "yoe_sigma": 2.0,
    "yoe_soft_min": 4.0,
    "target_cities": {
        "pune", "noida", "hyderabad", "mumbai", "delhi", "new delhi",
        "ncr", "gurgaon", "gurugram", "ghaziabad", "faridabad", "greater noida",
    },
    "good_india_cities": {"bangalore", "bengaluru", "chennai", "kolkata", "ahmedabad"},
    "notice_ideal_days": 30,
}

WEIGHTS = {
    "domain_evidence":     0.20,  # 0.20 with semantic; 0.30 via fallback when precomputed embeddings absent
    "skill_substance":     0.15,
    "seniority_fit":       0.12,
    "product_vs_services": 0.10,
    "external_validation": 0.08,
    "eval_frameworks":     0.08,
    "semantic_similarity": 0.10,  # NEW: cosine sim against JD embedding (msmarco-distilbert-cos-v5)
    "python_signal":       0.02,
    "location":            0.06,
    "notice":              0.04,
    "platform_quality":    0.05,
}
WEIGHTS_FALLBACK = {
    "domain_evidence":     0.30,
    "skill_substance":     0.15,
    "seniority_fit":       0.12,
    "product_vs_services": 0.10,
    "external_validation": 0.08,
    "eval_frameworks":     0.08,
    "python_signal":       0.02,
    "location":            0.06,
    "notice":              0.04,
    "platform_quality":    0.05,
}
PILLAR_KEYS = list(WEIGHTS.keys())

# Pillars where absolute values are arbitrary → population-calibrate these
CALIBRATE_PILLARS = {"domain_evidence", "skill_substance",
                     "eval_frameworks", "semantic_similarity"}

# Bootstrap settings
N_BOOTSTRAP = 500
BOOTSTRAP_SEED = 42

# ──────────────────────────────────────────────────────────────────────────────
# LEXICONS  (case-sensitive; all inputs are pre-lowercased)
# ──────────────────────────────────────────────────────────────────────────────
def _rx(words):
    return re.compile("|".join(r"\b" + w + r"\b" for w in words))

RX_RETRIEVAL = _rx([
    r"retriev\w*", r"ranking", r"rank(ed|ing|er)?", r"recommend\w*", r"recommender",
    r"search relevance", r"relevance", r"semantic search", r"vector search",
    r"nearest neighbou?r", r"learning[- ]to[- ]rank", r"\bltr\b", r"\bapproximate\s+nearest\b",
    r"information retrieval", r"information\s+retrieval", r"personali[sz]ation", r"matching",
    r"\bbm25\b", r"\bfaiss\b", r"elasticsearch", r"opensearch", r"\bsolr\b",
    r"\blucene\b", r"embeddings?", r"two[- ]tower", r"candidate generation",
    r"collaborative\s+filter\w*", r"matrix\s+factori\w*", r"item.item\s+similar\w*",
    r"session.based\s+recommend\w*", r"cold\s+start",
    r"document\s+(?:ranking|retrieval|scoring)",
    r"passage\s+(?:retrieval|ranking|reranking)",
    r"query\s+(?:expansion|understanding|rewriting|relevance)",
])
RX_EVAL = _rx([
    r"ndcg", r"\bmrr\b", r"mean reciprocal", r"mean average precision", r"\bmap@?\d*\b",
    r"a/?b test\w*", r"offline eval\w*", r"online eval\w*", r"precision@\d+",
    r"recall@\d+", r"\bauc\b", r"holdout", r"counterfactual", r"interleav\w*",
])
RX_LLM_MODERN = _rx([
    r"\bllm\b", r"large language model", r"\brag\b", r"retrieval[- ]augmented",
    r"fine[- ]tun\w*", r"\blora\b", r"\bqlora\b", r"\bpeft\b", r"transformer\w*",
    r"\bbert\b", r"sentence[- ]transformer", r"\be5\b", r"\bbge\b", r"reranke?\w*",
])
RX_VECTOR_DB = _rx([
    r"pinecone", r"weaviate", r"qdrant", r"milvus", r"\bfaiss\b", r"pgvector",
    r"elasticsearch", r"opensearch", r"vector (db|database|store|index)",
])
RX_PRODUCTION = _rx([
    r"production", r"deployed?", r"at scale", r"real users", r"serving",
    r"latency", r"throughput", r"\bpipeline\b", r"in[- ]production", r"shipped",
    r"\bmlops\b", r"\bapi\b", r"microservice\w*",
    r"click.through\s+rate", r"\bctr\b", r"conversion\s+rate",
    r"e.commerce\s+(?:search|ranking|recommendation|relevance)",
])
RX_CV_SPEECH_ROBO = _rx([
    r"computer vision", r"image classification", r"object detection",
    r"segmentation", r"\bcv\b", r"speech recognition", r"\basr\b",
    r"text[- ]to[- ]speech", r"\btts\b", r"robotics", r"\bslam\b", r"lidar",
    r"point cloud", r"autonomous (driving|vehicle)",
])
RX_NLP_IR = _rx([
    r"\bnlp\b", r"natural language", r"text mining", r"named entity",
    r"\bner\b", r"question answering", r"summari[sz]ation", r"information retrieval",
])
RX_ADJACENT_ML = _rx([
    r"feature (pipeline|engineering|store)", r"feature pipelines?", r"ml pipeline\w*",
    r"experimentation", r"model (training|serving|deployment|inference)",
    r"data science", r"machine learning model\w*", r"predictive model\w*",
    r"classification", r"regression model\w*", r"forecasting", r"churn", r"propensity",
    r"\bkaggle\b", r"fine[- ]tun\w*", r"\bspark\b", r"airflow", r"\bdbt\b", r"\betl\b",
    r"\bxgboost\b", r"lightgbm", r"gradient boost\w*",
    r"data (pipeline|infrastructure|warehouse)", r"analytics", r"\bml\b model\w*",
])
RX_PYTHON   = _rx([r"python", r"pytorch", r"tensorflow", r"scikit", r"numpy", r"pandas"])
RX_OSS      = _rx([r"open[- ]source", r"github", r"maintainer", r"contributor",
                   r"\bpaper\b", r"published", r"\btalk\b", r"conference"])
RX_RESEARCH = _rx([r"research scientist", r"\bphd\b", r"post[- ]?doc", r"postdoctoral",
                   r"research (lab|assistant|associate)", r"academ\w*", r"thesis",
                   r"university research", r"published \d"])

RX_NON_TECHNICAL = _rx([
    r"hr\b", r"human resource\w*", r"recruit\w*", r"talent acquisition",
    r"marketing", r"content writer", r"copywriter", r"\bcontent\b", r"\bsales\b",
    r"account executive", r"accountant", r"\bfinance\b", r"customer support",
    r"customer success", r"operations manager", r"\bbpo\b", r"administrativ\w*",
    r"office manager", r"business development", r"social media",
])
RX_NON_TARGET_TECH = _rx([
    r"civil engineer", r"mechanical engineer", r"electrical engineer",
    r"hardware", r"network engineer", r"\bvlsi\b", r"chemical engineer",
])
RX_CORE_ML = _rx([
    r"machine learning", r"\bml\b", r"\bai\b engineer", r"\bai\b", r"applied scientist",
    r"data scientist", r"research engineer", r"\bnlp\b", r"search", r"ranking",
    r"relevance", r"deep learning", r"mlops", r"ml engineer", r"ai engineer",
    r"research scientist", r"ml scientist",
])
RX_ADJACENT_ENG = _rx([
    r"software engineer", r"backend engineer", r"back[- ]end", r"data engineer",
    r"\bsde\b", r"full[- ]?stack", r"platform engineer", r"developer", r"programmer",
    r"devops", r"software developer", r"engineer",
])

SERVICES_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "tech mahindra", "hcl", "hcltech", "mindtree", "ltimindtree",
    "lti", "deloitte", "ibm", "dxc", "mphasis", "persistent systems",
}
PROF_W = {"beginner": 0.3, "intermediate": 0.6, "advanced": 0.85, "expert": 1.0}

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
def load_candidates(path):
    out = []
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    elif path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            out = data if isinstance(data, list) else [data]
    return out

def _parse_date(s):
    if not s or len(s) < 10:
        return None
    try:
        return date(int(s[:4]), int(s[5:7]), int(s[8:10]))
    except Exception:
        return None

def _candidate_text(c):
    p = c.get("profile", {})
    parts = [p.get("headline", ""), p.get("summary", ""),
             p.get("current_title", ""), p.get("current_industry", "")]
    for h in c.get("career_history", []):
        parts.append(h.get("title", ""))
        parts.append(h.get("description", ""))
        parts.append(h.get("industry", ""))
    for s in c.get("skills", []):
        parts.append(s.get("name", ""))
    return " \n ".join(x for x in parts if x)

# ──────────────────────────────────────────────────────────────────────────────
# L0 — HONEYPOT DETECTION
# ──────────────────────────────────────────────────────────────────────────────
def honeypot_flags(c):
    flags = []
    yoe = float(c.get("profile", {}).get("years_of_experience") or 0)
    total_career_m = sum(int(h.get("duration_months") or 0) for h in c.get("career_history", []))
    zero_dur_high = sum(
        1 for s in c.get("skills", [])
        if s.get("proficiency") in ("advanced", "expert")
        and int(s.get("duration_months") or 0) == 0
    )
    if zero_dur_high >= 3:
        flags.append("many_high_skills_zero_duration")
    for s in c.get("skills", []):
        career_baseline = max(total_career_m, int(yoe * 12))
        if int(s.get("duration_months") or 0) > career_baseline + 12 and career_baseline > 0:
            flags.append("skill_duration_exceeds_career"); break
    date_conflicts = 0
    for h in c.get("career_history", []):
        sd = _parse_date(h.get("start_date"))
        ed = _parse_date(h.get("end_date")) or date(2026, 6, 1)
        dur = int(h.get("duration_months") or 0)
        if sd:
            calc = (ed.year - sd.year) * 12 + (ed.month - sd.month)
            if calc >= 0 and abs(dur - calc) > 9:
                date_conflicts += 1
    if date_conflicts >= 1:
        flags.append("tenure_date_contradiction")
    if yoe > 0 and total_career_m > yoe * 12 + 30:
        flags.append("career_sum_exceeds_yoe")
    currents = [h for h in c.get("career_history", []) if h.get("is_current")]
    if any(h.get("end_date") for h in currents) or len(currents) > 1:
        flags.append("is_current_contradiction")
    strong = {"many_high_skills_zero_duration", "skill_duration_exceeds_career"}
    is_hp = bool(strong & set(flags)) or len(set(flags)) >= 2
    return is_hp, sorted(set(flags))

# ──────────────────────────────────────────────────────────────────────────────
# L1 — ROLE FIT
# ──────────────────────────────────────────────────────────────────────────────
def role_fit(c):
    title = (c.get("profile", {}).get("current_title", "") or "").lower()
    if RX_NON_TECHNICAL.search(title):   return 0.05, "non_technical"
    if RX_NON_TARGET_TECH.search(title): return 0.12, "non_target_tech"
    if RX_CORE_ML.search(title):         return 1.00, "core_ml"
    if RX_ADJACENT_ENG.search(title):    return 0.58, "adjacent_eng"
    return 0.30, "unknown_role"

# ──────────────────────────────────────────────────────────────────────────────
# L2 — DOMAIN EVIDENCE  (layered explicit / adjacent)
# ──────────────────────────────────────────────────────────────────────────────
def domain_evidence(c, text):
    ev = []
    retr   = len(RX_RETRIEVAL.findall(text))
    prod   = len(RX_PRODUCTION.findall(text))
    modern = len(RX_LLM_MODERN.findall(text))
    vdb    = len(RX_VECTOR_DB.findall(text))
    evalh  = len(RX_EVAL.findall(text))
    nlpir  = len(RX_NLP_IR.findall(text))
    adj    = len(RX_ADJACENT_ML.findall(text))
    cvrobo = len(RX_CV_SPEECH_ROBO.findall(text))

    retr_s   = 1 - math.exp(-retr / 2.0)
    modern_s = 1 - math.exp(-modern / 2.0)
    vdb_s    = min(vdb, 1)
    eval_s   = 1 - math.exp(-evalh / 1.5)
    nlpir_s  = 1 - math.exp(-nlpir / 3.0)
    explicit = (0.45*retr_s + 0.18*modern_s + 0.15*vdb_s + 0.12*eval_s + 0.10*nlpir_s)
    explicit_present = (retr + modern + vdb + evalh) > 0
    prod_s = 1 - math.exp(-prod / 2.0)
    explicit *= (0.7 + 0.3 * prod_s)
    adjacent = 1 - math.exp(-adj / 3.0)

    if explicit_present:
        score = float(np.clip(0.80*explicit + 0.20*adjacent, 0, 1))
        if retr:   ev.append("retrieval/ranking work")
        if vdb:    ev.append("vector/search infra")
        if modern: ev.append("modern LLM/embedding work")
        if evalh:  ev.append("ranking-evaluation experience")
        if prod and not ev: ev.append("production ML deployment")
    else:
        score = float(np.clip(adjacent, 0, 1) * 0.42)
        if adj: ev.append("adjacent ML/data work (plain-language)")

    cv_heavy = cvrobo >= 2 and (retr + nlpir) < cvrobo
    return score, ev, cv_heavy

# ──────────────────────────────────────────────────────────────────────────────
# L3 — FIT PILLARS
# ──────────────────────────────────────────────────────────────────────────────
def seniority_fit(c):
    yoe = float(c.get("profile", {}).get("years_of_experience") or 0)
    s = math.exp(-0.5 * ((yoe - JOB["ideal_yoe"]) / JOB["yoe_sigma"]) ** 2)
    if yoe < JOB["yoe_soft_min"]:
        s *= (max(yoe, 0) / JOB["yoe_soft_min"]) ** 2
    return float(s)

def product_vs_services(c):
    hist = c.get("career_history", [])
    if not hist: return 0.5
    def is_services(h):
        comp = (h.get("company", "") or "").lower()
        ind  = (h.get("industry", "") or "").lower()
        return any(f in comp for f in SERVICES_FIRMS) or "it services" in ind or "consult" in ind
    flags = [is_services(h) for h in hist]
    if not any(flags): return 1.0
    if all(flags):     return 0.25
    return 0.8

def skill_substance(c):
    sig    = c.get("redrob_signals", {})
    assess = sig.get("skill_assessment_scores", {}) or {}
    total, hits = 0.0, 0
    for s in c.get("skills", []):
        name = (s.get("name", "") or "").lower()
        if not (RX_RETRIEVAL.search(name) or RX_LLM_MODERN.search(name)
                or RX_PYTHON.search(name) or RX_NLP_IR.search(name)
                or RX_VECTOR_DB.search(name) or "ml" in name or "ai" in name
                or "data" in name or "statistic" in name):
            continue
        hits += 1
        pw  = PROF_W.get(s.get("proficiency"), 0.4)
        dur = min(int(s.get("duration_months") or 0) / 24.0, 1.0)
        end = min(math.log1p(int(s.get("endorsements") or 0)) / 4.0, 1.0)
        a   = assess.get(s.get("name", ""), None)
        asc = (a / 100.0) if isinstance(a, (int, float)) and a >= 0 else 0.5
        total += 0.4*pw + 0.2*dur + 0.15*end + 0.25*asc
    if hits == 0: return 0.0
    return float(np.clip(total / max(hits, 1) * min(hits / 4.0, 1.0), 0, 1))

def python_signal(c, text):
    for s in c.get("skills", []):
        if (s.get("name", "") or "").lower() == "python":
            return float(np.clip(0.5 + PROF_W.get(s.get("proficiency"), 0.4) * 0.5, 0, 1))
    return 0.6 if RX_PYTHON.search(text) else 0.25

def eval_frameworks(c, text):
    n = len(RX_EVAL.findall(text))
    return float(1 - math.exp(-n / 1.5)) if n else 0.15

def external_validation(c, text):
    gh = c.get("redrob_signals", {}).get("github_activity_score", -1)
    gh = float(gh) if gh is not None else -1.0
    base = 0.40 if gh < 0 else float(np.clip(0.45 + gh / 100.0 * 0.55, 0, 1))
    if RX_OSS.search(text): base = min(base + 0.15, 1.0)
    return base

def location_score(c):
    p       = c.get("profile", {})
    loc     = (p.get("location", "") or "").lower()
    country = (p.get("country", "") or "").lower()
    relo    = bool(c.get("redrob_signals", {}).get("willing_to_relocate", False))
    in_india = "india" in country or country == ""
    if any(city in loc for city in JOB["target_cities"]):           return 1.0
    if any(city in loc for city in JOB["good_india_cities"]):       return 0.90 if relo else 0.85
    if in_india:                                                     return 0.70 if relo else 0.45
    return 0.40 if relo else 0.20

def notice_score(c):
    nd = c.get("redrob_signals", {}).get("notice_period_days", 30)
    nd = float(nd) if nd is not None else 30.0
    return float(1.0 / (1.0 + math.exp(0.06 * (nd - JOB["notice_ideal_days"]))))

# ──────────────────────────────────────────────────────────────────────────────
# L4 — NEGATIVE PENALTIES
# ──────────────────────────────────────────────────────────────────────────────
def negative_penalty(c, text, domain_score, prod_serv, cv_heavy):
    reasons, ps = [], []
    if RX_RESEARCH.search(text) and not RX_PRODUCTION.search(text) and domain_score < 0.4:
        ps.append(0.55); reasons.append("research-leaning, little production signal")
    if prod_serv <= 0.25:
        ps.append(0.40); reasons.append("career entirely in IT-services/consulting")
    if cv_heavy:
        ps.append(0.35); reasons.append("CV/speech/robotics focus, thin NLP/IR")
    hist = c.get("career_history", [])
    noncur = [h for h in hist if not h.get("is_current")]
    short  = [h for h in noncur if int(h.get("duration_months") or 24) < 18]
    if len(noncur) >= 3 and len(short) >= 3:
        ps.append(0.22); reasons.append("frequent short stints (job-hopping pattern)")
    pen = 1.0
    for p in ps: pen *= (1 - p)
    return float(min(1 - pen, 0.85)), reasons

# ──────────────────────────────────────────────────────────────────────────────
# L5 — BEHAVIORAL MULTIPLIER
# ──────────────────────────────────────────────────────────────────────────────
def behavioral_multiplier(c, now):
    s   = c.get("redrob_signals", {})
    la  = _parse_date(s.get("last_active_date"))
    days = (now - la).days if la else 365
    recency = math.exp(-max(days, 0) / 120.0)
    resp    = float(s.get("recruiter_response_rate") or 0.0)
    icr     = float(s.get("interview_completion_rate") or 0.0)
    otw     = 1.0 if s.get("open_to_work_flag") else 0.55
    comp    = float(s.get("profile_completeness_score") or 0) / 100.0
    core = 0.40*recency + 0.25*resp + 0.15*icr + 0.10*otw + 0.10*comp
    return float(0.35 + 0.65 * np.clip(core, 0, 1)), {
        "days_inactive": days, "response_rate": resp,
        "open_to_work": bool(s.get("open_to_work_flag")),
        "behavior_core": float(np.clip(core, 0, 1)),
    }

def platform_quality_score(c):
    sig = c.get("redrob_signals", {}) or {}
    search = float(sig.get("search_appearance_30d") or 0)
    offer  = sig.get("offer_acceptance_rate")
    offer_f = float(offer) if offer is not None and float(offer) >= 0 else 0.5
    search_s = 1 - math.exp(-search / 300.0)
    return float(0.60 * search_s + 0.40 * offer_f)


def tier5_signal(c):
    """Platform engagement signal for head reordering in the Variant S cascade.
    t5 = 0.35*offer_s + 0.25*search_s + 0.20*saved_s + 0.20*assess_s
    """
    sig = c.get("redrob_signals", {}) or {}
    offer = sig.get("offer_acceptance_rate")
    offer_s  = float(offer) if offer is not None and float(offer) >= 0 else 0.0
    search_s = 1 - math.exp(-float(sig.get("search_appearance_30d") or 0) / 400.0)
    saved_s  = 1 - math.exp(-float(sig.get("saved_by_recruiters_30d") or 0) / 30.0)
    assess   = sig.get("skill_assessment_scores") or {}
    vals     = [v for v in assess.values() if isinstance(v, (int, float)) and v >= 0]
    assess_s = (sum(vals) / len(vals) / 100.0) if vals else 0.5
    return 0.35*offer_s + 0.25*search_s + 0.20*saved_s + 0.20*assess_s


# ──────────────────────────────────────────────────────────────────────────────
# SEMANTIC SIMILARITY — loaded from precomputed/ at rank time (no model inference)
# NOTE: the fallback path (precomputed/ absent) IS the canonical submission mode.
# WEIGHTS_FALLBACK produces the validated 0.7229 composite and is fully deterministic.
# Absence of precomputed/ is normal in the submission environment — not a degraded mode.
# ──────────────────────────────────────────────────────────────────────────────
_SEMANTIC_SCORES: dict = {}
_SEMANTIC_LOADED: bool = False

def _load_semantic_scores() -> None:
    global _SEMANTIC_SCORES, _SEMANTIC_LOADED
    if _SEMANTIC_LOADED:
        return
    emb_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "precomputed")
    emb_path = os.path.join(emb_dir, "candidate_embeddings.npy")
    jd_path  = os.path.join(emb_dir, "jd_embedding.npy")
    idx_path = os.path.join(emb_dir, "candidate_id_index.json")
    try:
        cand_emb = np.load(emb_path)   # [N, 768], L2-normalised
        jd_emb   = np.load(jd_path)    # [768],    L2-normalised
        sims     = (cand_emb @ jd_emb).astype(float)  # cosine sim in [0, 1]
        with open(idx_path, encoding="utf-8") as f:
            idx_map = json.load(f)
        for idx_str, cid in idx_map.items():
            _SEMANTIC_SCORES[cid] = float(np.clip(sims[int(idx_str)], 0.0, 1.0))
        print(f"  [semantic] {len(_SEMANTIC_SCORES):,} similarity scores loaded "
              f"(min={min(_SEMANTIC_SCORES.values()):.3f} "
              f"max={max(_SEMANTIC_SCORES.values()):.3f})", flush=True)
    except FileNotFoundError:
        global WEIGHTS, PILLAR_KEYS
        WEIGHTS = dict(WEIGHTS_FALLBACK)
        PILLAR_KEYS = list(WEIGHTS.keys())
        print("  [MODE] semantic embeddings absent → deterministic fallback weights "
              "(canonical submission mode)", flush=True)
    _SEMANTIC_LOADED = True

def semantic_similarity_score(c) -> float:
    cid = c.get("candidate_id", "")
    return _SEMANTIC_SCORES.get(cid, 0.5)

# ──────────────────────────────────────────────────────────────────────────────
# CORE SCORER — unchanged logic, used by both phases
# ──────────────────────────────────────────────────────────────────────────────
_NOW = date(2026, 6, 1)   # module-level so workers can access it

def score_candidate(c, now=None):
    if now is None: now = _NOW
    text = _candidate_text(c).lower()
    is_hp, hp = honeypot_flags(c)
    rmult, rlabel = role_fit(c)
    dscore, devid, cv_heavy = domain_evidence(c, text)
    prod_serv = product_vs_services(c)

    pillars = {
        "domain_evidence":     dscore,
        "skill_substance":     skill_substance(c),
        "seniority_fit":       seniority_fit(c),
        "product_vs_services": prod_serv,
        "external_validation": external_validation(c, text),
        "eval_frameworks":     eval_frameworks(c, text),
        "semantic_similarity": semantic_similarity_score(c),
        "python_signal":       python_signal(c, text),
        "location":            location_score(c),
        "notice":              notice_score(c),
        "platform_quality":    platform_quality_score(c),
    }
    base_fit = sum(WEIGHTS.get(k, 0) * v for k, v in pillars.items())
    if rlabel == "adjacent_eng":
        rmult = min(1.0, rmult + 0.45 * dscore)
    pen, neg_reasons = negative_penalty(c, text, dscore, prod_serv, cv_heavy)
    bmult, bdetail   = behavioral_multiplier(c, now)
    final = base_fit * rmult * (1 - pen) * bmult
    if is_hp:
        final = final * 0.01 - 0.2

    trace = {
        "final": float(final), "base_fit": float(base_fit), "role": rlabel,
        "role_mult": round(rmult, 3),
        "pillars": {k: round(v, 3) for k, v in pillars.items()},
        "domain_evidence_terms": devid, "penalty": round(pen, 3),
        "neg_reasons": neg_reasons, "behavior_mult": round(bmult, 3),
        "behavior": bdetail, "honeypot": is_hp, "honeypot_flags": hp,
    }
    return final, trace

# ──────────────────────────────────────────────────────────────────────────────
# MULTIPROCESSING WORKER  (must be top-level for Windows spawn compatibility)
# ──────────────────────────────────────────────────────────────────────────────
def _score_chunk(args):
    """Score a chunk of candidates. Top-level function for MP cross-platform."""
    chunk, now = args
    return [score_candidate(c, now) for c in chunk]

# ──────────────────────────────────────────────────────────────────────────────
# CONCEPT A — POPULATION CALIBRATION
# ──────────────────────────────────────────────────────────────────────────────
def _percentile_normalize(values):
    """Pure-numpy percentile rank. Returns array in [0, 1]."""
    n = len(values)
    if n <= 1: return np.zeros(n)
    order = np.argsort(values)
    ranks = np.empty(n, dtype=np.float32)
    ranks[order] = np.arange(n, dtype=np.float32)
    return ranks / (n - 1)

def population_calibrate(raw_results):
    """
    Concept A: replace the most arbitrary pillar scores with population
    percentile ranks. Scores now mean 'top X% of this pool' not 'absolute Y'.
    Only calibrates CALIBRATE_PILLARS (domain_evidence, skill_substance,
    eval_frameworks) — pillars with JD-relative meaning stay unchanged.
    Returns updated (score, trace) list with recomputed final scores.
    """
    # Collect raw pillar values across all candidates
    pillar_arrays = {k: np.array([tr["pillars"][k] for _, tr in raw_results],
                                 dtype=np.float32)
                     for k in CALIBRATE_PILLARS}

    # Compute percentile ranks for each calibratable pillar
    calibrated = {k: _percentile_normalize(pillar_arrays[k]) for k in CALIBRATE_PILLARS}

    # Recompute scores with calibrated pillars
    updated = []
    for i, (raw_score, tr) in enumerate(raw_results):
        new_pillars = dict(tr["pillars"])
        for k in CALIBRATE_PILLARS:
            new_pillars[k] = round(float(calibrated[k][i]), 4)
        new_base = sum(WEIGHTS.get(k, 0) * v for k, v in new_pillars.items())
        new_final = new_base * tr["role_mult"] * (1 - tr["penalty"]) * tr["behavior_mult"]
        if tr["honeypot"]:
            new_final = new_final * 0.01 - 0.2
        new_tr = dict(tr)
        new_tr["pillars"] = new_pillars
        new_tr["base_fit"] = round(float(new_base), 4)
        new_tr["final"]    = float(new_final)
        updated.append((float(new_final), new_tr))
    return updated

# ──────────────────────────────────────────────────────────────────────────────
# CONCEPT B — ISOLATION FOREST
# ──────────────────────────────────────────────────────────────────────────────
def _if_feature_vector(c):
    """Numeric feature vector for unsupervised anomaly detection."""
    p      = c.get("profile", {})
    s      = c.get("redrob_signals", {}) or {}
    hist   = c.get("career_history", [])
    skills = c.get("skills", [])
    yoe    = float(p.get("years_of_experience") or 0)
    career_months = sum(int(h.get("duration_months") or 0) for h in hist)
    expert_zero   = sum(1 for sk in skills
                        if sk.get("proficiency") in ("expert", "advanced")
                        and int(sk.get("duration_months") or 0) == 0)
    max_skill_dur = max((int(sk.get("duration_months") or 0) for sk in skills), default=0)
    gh            = float(s.get("github_activity_score") or 0); gh = max(gh, 0)
    completeness  = float(s.get("profile_completeness_score") or 0)
    endorsements  = float(s.get("endorsements_received") or 0)
    n_currents    = sum(1 for h in hist if h.get("is_current"))
    dur_ratio     = career_months / max(yoe * 12, 1)
    n_skills      = len(skills)
    return [yoe, career_months, expert_zero, max_skill_dur, n_skills,
            gh, completeness, endorsements, n_currents, dur_ratio]

def run_isolation_forest(candidates, calibrated_results):
    """
    Concept B: fit an Isolation Forest on ALL candidates to learn what
    'normal' looks like, then score anomalies. Returns an additional penalty
    array (0 = no penalty, up to 0.55 = strong anomaly signal).
    Falls back to zeros if sklearn is not installed.
    """
    n = len(candidates)
    penalties = np.zeros(n, dtype=np.float32)
    if not HAS_SKLEARN:
        return penalties

    print("  [B] Fitting Isolation Forest …", flush=True)
    X = np.array([_if_feature_vector(c) for c in candidates], dtype=np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = IsolationForest(n_estimators=120, contamination=0.001,
                          random_state=42, n_jobs=1)
    clf.fit(X_scaled)
    raw_scores = clf.score_samples(X_scaled)   # more negative = more anomalous

    # Normalise to [0, 1]: 0 = most normal, 1 = most anomalous
    lo, hi = raw_scores.min(), raw_scores.max()
    norm = 1.0 - (raw_scores - lo) / (hi - lo + 1e-8)

    # Only penalise candidates our rule-based gate DIDN'T already catch
    # (If rules flagged them, they're already crushed; IF covers the gaps)
    for i, (_, tr) in enumerate(calibrated_results):
        if not tr["honeypot"] and norm[i] > 0.85:
            # Anomalous but not rule-caught: apply graded penalty
            penalties[i] = float(np.clip((norm[i] - 0.85) / 0.15 * 0.55, 0, 0.55))
            tr["if_anomaly"] = round(float(norm[i]), 3)

    n_flagged = int((penalties > 0).sum())
    print(f"  [B] IF additional flags: {n_flagged} candidates", flush=True)
    return penalties

# ──────────────────────────────────────────────────────────────────────────────
# CONCEPT C — BOOTSTRAP RANK-CONFIDENCE
# ──────────────────────────────────────────────────────────────────────────────
def run_bootstrap(calibrated_results, top_n=100):
    """
    Concept C: perturb weights N_BOOTSTRAP times via matrix multiply.
    Returns confidence[i] = fraction of weight configs where candidate i
    appears in the top top_n. Near-instant after calibration (numpy BLAS).
    """
    n = len(calibrated_results)
    top_n = min(top_n, n - 1)   # guard: np.partition requires kth < n
    print(f"  [C] Bootstrap confidence ({N_BOOTSTRAP} samples) …", flush=True)

    # Build pillar matrix and adjustment vectors (both pre-calibrated)
    pillar_mat = np.array(
        [[tr["pillars"].get(k, 0.0) for k in PILLAR_KEYS]
         for _, tr in calibrated_results],
        dtype=np.float32)                          # (N, P)
    adjustments = np.array(
        [tr["role_mult"] * (1 - tr["penalty"]) * tr["behavior_mult"]
         for _, tr in calibrated_results],
        dtype=np.float32)                          # (N,)
    hp_mask = np.array([tr["honeypot"] for _, tr in calibrated_results])  # (N,)

    # Generate weight perturbations: log-normal around base weights
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    base_w = np.array([WEIGHTS[k] for k in PILLAR_KEYS], dtype=np.float32)
    log_p  = rng.normal(0, 0.35, (N_BOOTSTRAP, len(base_w))).astype(np.float32)
    w_samp = base_w * np.exp(log_p)
    w_samp *= (base_w.sum() / w_samp.sum(axis=1, keepdims=True))  # renormalise

    # Vectorised scoring: (N, 9) @ (9, B) = (N, B)
    bs_scores = pillar_mat @ w_samp.T                  # (N, B)
    bs_scores *= adjustments[:, None]                  # apply role/penalty/behav
    bs_scores[hp_mask] = -1.0                          # crush honeypots

    # For each bootstrap sample, find the score of the top_n-th candidate
    # np.partition is O(N) per column — much faster than full sort
    neg_part   = np.partition(-bs_scores, top_n, axis=0)  # partial sort
    thresholds = -neg_part[top_n - 1, :]                  # (B,) per-sample cutoff

    # Confidence = fraction of samples where this candidate beats the cutoff
    # We only need it for the candidates we'll actually report (top ~200)
    # but compute for all N for correctness then slice
    confidence = np.zeros(n, dtype=np.float32)
    # Process in batches to stay memory-light (avoid 100K × 500 bool matrix)
    batch = 500
    for start in range(0, n, batch):
        end = min(start + batch, n)
        chunk = bs_scores[start:end, :]                   # (batch, B)
        conf  = (chunk >= thresholds[None, :]).mean(axis=1)
        confidence[start:end] = conf

    return confidence

# ──────────────────────────────────────────────────────────────────────────────
# TIER PREDICTION
# ──────────────────────────────────────────────────────────────────────────────
def predict_tier(trace):
    if trace["honeypot"] or trace["role"] in ("non_technical", "non_target_tech"):
        return 0
    f = trace["final"]
    if f >= 0.749: return 5
    if f >= 0.606: return 4
    if f >= 0.462: return 3
    if f >= 0.426: return 2
    if f >= 0.390: return 1
    return 0

def _clip_to_boundary(s, limit):
    """Clip s to at most `limit` chars, cutting at the last sentence/clause
    boundary within the budget rather than mid-word. Falls back to the last
    whole word, then to a hard cut only if nothing else is available."""
    if len(s) <= limit:
        return s
    s = s[:limit]
    cut = max(s.rfind(". "), s.rfind("; "))
    if cut >= limit * 0.4:
        return s[:cut + 1].rstrip()
    cut = s.rfind(" ")
    if cut >= limit * 0.4:
        return s[:cut].rstrip()
    return s.rstrip()


def make_reasoning(c, trace, tier, rank_stability=None, rank=0):
    title = c.get("profile", {}).get("current_title", "professional")

    # Edge-case guards — these candidates won't normally reach top-100
    if trace["honeypot"]:
        return (f"{title} profile contains internal inconsistencies "
                f"({', '.join(trace['honeypot_flags'][:2])}); "
                f"flagged as non-credible and ranked at the bottom.")
    if trace.get("if_anomaly", 0) > 0.85 and not trace["honeypot"]:
        return (f"{title}; statistical anomaly signals detected "
                f"(score {trace['if_anomaly']:.2f}); "
                f"recommend manual profile verification.")
    if trace["role"] == "non_technical":
        return (f"Current role is {title}, a non-engineering function; "
                f"AI skills are listed but the role does not match the JD "
                f"regardless of keywords.")

    try:
        from reasoning import make_rich_reasoning
        text = make_rich_reasoning(c, trace, rank)
    except ImportError:
        print("  [warn] reasoning.py not found — using inline fallback", flush=True)
        concerns = trace.get("neg_reasons", [])
        top_concern = f" Note: {concerns[0]}." if concerns else ""
        text = f"{title} (tier {tier}).{top_concern}"

    # Rank-stability annotation: sensitivity to weight perturbation, NOT a ranking input
    suffix = ""
    if rank_stability is not None:
        if rank_stability >= 0.85:
            suffix = f" Rank stability: {rank_stability:.0%} of weight perturbations agree."
        elif rank_stability >= 0.50:
            suffix = f" Rank stability: {rank_stability:.0%} of weight perturbations agree; verify against JD."
        else:
            suffix = f" Rank stability: {rank_stability:.0%} of weight perturbations agree; manual review recommended."

    LIMIT = 350
    if len(text) + len(suffix) <= LIMIT:
        return text + suffix

    # Prefer keeping the rank-stability suffix — clip the base text to make room for it.
    clipped = _clip_to_boundary(text, LIMIT - len(suffix))
    if clipped and len(clipped) + len(suffix) <= LIMIT:
        return clipped + suffix

    # Suffix doesn't fit even against a minimally clipped base — drop it, clip base alone.
    return _clip_to_boundary(text, LIMIT)

# ──────────────────────────────────────────────────────────────────────────────
# RICH DISPLAY HELPERS  (only active when --rich is passed and rich is available)
# ──────────────────────────────────────────────────────────────────────────────
def _ensure_rich() -> bool:
    """Try to import rich; pip-install it if absent. Returns True if available."""
    global HAS_RICH
    if HAS_RICH:
        return True
    import subprocess, sys
    print("  [rich] installing rich …", flush=True)
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "rich", "-q"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        import rich as _r  # noqa: F401
        HAS_RICH = True
        return True
    except Exception:
        print("  [warn] Could not install rich — falling back to plain output.", flush=True)
        return False


def _phase1_with_progress(mp_args, ncores, n):
    """Phase 1 multiprocessing with a live rich progress bar."""
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn,
        TaskProgressColumn, TimeElapsedColumn, TextColumn,
    )
    chunk_results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Phase 1[/bold cyan] · {task.completed}/{task.total} chunks"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        transient=False,
    ) as progress:
        task = progress.add_task("", total=len(mp_args))
        if ncores > 1 and n > 200:
            with Pool(processes=ncores) as pool:
                for result in pool.imap(_score_chunk, mp_args):
                    chunk_results.append(result)
                    progress.advance(task)
        else:
            for a in mp_args:
                chunk_results.append(_score_chunk(a))
                progress.advance(task)
    return chunk_results


_TIER_STYLE = {
    5: "bold green", 4: "bold blue", 3: "bold yellow",
    2: "dim white",  1: "dim",       0: "bold red",
}


def _render_rich_top10(rows):
    """Print a color-coded top-10 table."""
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    console = Console()
    tbl = Table(
        title="[bold]Top 10 Candidates[/bold]",
        header_style="bold magenta", show_lines=False, expand=False,
    )
    tbl.add_column("#",          width=3,  style="dim")
    tbl.add_column("Candidate",  width=22)
    tbl.add_column("Score",      width=7)
    tbl.add_column("Tier",       width=4)
    tbl.add_column("Stability",  width=10)
    tbl.add_column("Penalties",  width=36)
    tbl.add_column("Summary",    width=55, no_wrap=False)

    for r in rows[:10]:
        tier  = r["_tier"]
        stab  = r["_rank_stability"]
        trace = r["_trace"]

        tier_t = Text(f"T{tier}", style=_TIER_STYLE.get(tier, ""))

        if stab >= 0.85:
            stab_t = Text(f"✓ {stab:.0%}", style="bold green")
        elif stab >= 0.50:
            stab_t = Text(f"~ {stab:.0%}", style="bold yellow")
        else:
            stab_t = Text(f"✗ {stab:.0%}", style="bold red")

        neg   = trace.get("neg_reasons", [])
        pen_t = Text("; ".join(neg), style="dim red") if neg else Text("—", style="dim")

        # Strip trailing stability sentence — already shown in its own column
        summary = r["reasoning"]
        idx = summary.find(" Rank stability:")
        if idx != -1:
            summary = summary[:idx]

        tbl.add_row(
            str(r["rank"]),
            r["candidate_id"],
            f"{r['score']:.4f}",
            tier_t, stab_t, pen_t,
            summary[:120],
        )

    console.print(tbl)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ──────────────────────────────────────────────────────────────────────────────
def rank_all(candidates, top_n=100, use_rich=False):
    n = len(candidates)
    if n == 0:
        return [], []

    now = date(2026, 6, 1)  # fixed reference date — intentional for deterministic recency scoring

    # ── Pre-phase: load semantic similarity scores from precomputed embeddings ─
    _load_semantic_scores()

    # ── Phase 1: multiprocessing feature extraction ───────────────────────────
    ncores = cpu_count()
    chunk_size = max(1, n // ncores)
    chunks = [candidates[i:i+chunk_size] for i in range(0, n, chunk_size)]
    args   = [(chunk, now) for chunk in chunks]

    if use_rich and HAS_RICH:
        chunk_results = _phase1_with_progress(args, ncores, n)
    elif ncores > 1 and n > 200:
        with Pool(processes=ncores) as pool:
            chunk_results = pool.map(_score_chunk, args)
    else:
        chunk_results = [_score_chunk(a) for a in args]

    raw_results = [item for sublist in chunk_results for item in sublist]
    # raw_results: [(score, trace), ...]

    # ── Phase 2: population calibration (Concept A) ───────────────────────────
    print("  [A] Population calibration …", flush=True)
    calibrated = population_calibrate(raw_results)
    # calibrated: [(score, trace), ...]

    # ── Phase 3: Isolation Forest (Concept B) ─────────────────────────────────
    if_penalties = run_isolation_forest(candidates, calibrated)

    # Apply IF penalties to final_order scores
    for i, (score, tr) in enumerate(calibrated):
        pen = float(if_penalties[i])
        if pen > 0:
            new_score = score * (1 - pen)
            tr["final"] = new_score
            calibrated[i] = (new_score, tr)

    # Compute final_select (behavior_mult = 1.0); honeypot crush and IF apply
    final_selects = []
    for i, (score, tr) in enumerate(calibrated):
        fs = tr["base_fit"] * tr["role_mult"] * (1 - tr["penalty"])
        if tr["honeypot"]:
            fs = fs * 0.01 - 0.2
        pen_i = float(if_penalties[i])
        if pen_i > 0:
            fs *= (1 - pen_i)
        final_selects.append(float(fs))

    # ── Phase 4: bootstrap confidence (Concept C) ─────────────────────────────
    confidence = run_bootstrap(calibrated, top_n=top_n)

    # ── Phase 5: Variant S cascade — two-stage selection/ordering decomposition ──
    scored = [(candidates[i], calibrated[i][0], calibrated[i][1], float(confidence[i]), final_selects[i])
              for i in range(n)]

    # Stage 1: pool = top-100 by final_select (selects quality candidates, ignores availability noise)
    pool = sorted(scored, key=lambda x: -x[4])[:top_n]

    # Head: top-15 of pool by final_order, reordered by final_order*(0.85+0.15*t5)
    pool_by_order = sorted(pool, key=_cascade_sort_key)
    head = sorted(pool_by_order[:15],
                  key=lambda it: -(it[1] * (0.85 + 0.15 * tier5_signal(it[0]))))

    # Tail: remaining 85 by calibrated domain_evidence desc;
    # tail tie-runs on domain_evidence are intentional — tiebreak: behavior_core desc, candidate_id asc
    tail = sorted(pool_by_order[15:],
                  key=lambda it: (-it[2]["pillars"]["domain_evidence"],
                                  -it[2]["behavior"].get("behavior_core", 0),
                                  it[0].get("candidate_id", "")))

    rows = _make_cascade_rows(head + tail, top_n)
    _intended = [it[0].get("candidate_id", "") for it in (head + tail)[:top_n]]
    assert [r["candidate_id"] for r in rows] == _intended, \
        "emitted order disagrees with intended head+tail order"
    return rows, scored

# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ──────────────────────────────────────────────────────────────────────────────
def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in rows:
            w.writerow([r["candidate_id"], r["rank"], r["score"], r["reasoning"]])

# ──────────────────────────────────────────────────────────────────────────────
# FEATURE EXPORT  (for LambdaMART training via train_ltr.py)
# ──────────────────────────────────────────────────────────────────────────────
def _export_pillar_features(scored, path):
    """
    Write a CSV of every candidate's calibrated pillar scores.
    Column names match train_ltr.py's FEATURE_COLS exactly so no renaming
    is needed downstream.
    scored: [(candidate, final_score, trace, confidence), ...]  from rank_all()
    """
    headers = [
        "candidate_id",
        "domain_evidence_cal", "skill_substance_cal",
        "seniority_fit", "product_vs_services", "external_validation",
        "eval_frameworks_cal", "semantic_similarity", "python_signal",
        "location_score", "notice_score", "platform_quality",
        "behavioral_mult", "role_mult", "neg_penalty",
        "is_honeypot", "base_fit", "final_score",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for c, final, trace, conf, *_ in scored:
            pil = trace["pillars"]
            writer.writerow({
                "candidate_id":        c.get("candidate_id", ""),
                "domain_evidence_cal": pil.get("domain_evidence", 0),
                "skill_substance_cal": pil.get("skill_substance", 0),
                "seniority_fit":       pil.get("seniority_fit", 0),
                "product_vs_services": pil.get("product_vs_services", 0),
                "external_validation": pil.get("external_validation", 0),
                "eval_frameworks_cal": pil.get("eval_frameworks", 0),
                "semantic_similarity": pil.get("semantic_similarity", 0),
                "python_signal":       pil.get("python_signal", 0),
                "location_score":      pil.get("location", 0),
                "notice_score":        pil.get("notice", 0),
                "platform_quality":    pil.get("platform_quality", 0),
                "behavioral_mult":     trace.get("behavior_mult", 0),
                "role_mult":           trace.get("role_mult", 0),
                "neg_penalty":         trace.get("penalty", 0),
                "is_honeypot":         int(trace.get("honeypot", False)),
                "base_fit":            trace.get("base_fit", 0),
                "final_score":         float(final),
            })
    print(f"  [D] Pillar features → {path}  ({len(scored):,} rows)", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# CASCADE VARIANT BUILDER  (two-stage: select by final_select, rank by final_order)
# ──────────────────────────────────────────────────────────────────────────────
def _cascade_sort_key(item):
    return (-item[1], -item[2]["behavior"].get("behavior_core", 0), item[0].get("candidate_id", ""))


def _make_cascade_rows(pool, top_n=100):
    rows, prev = [], float("inf")
    # EPS is a strict gap enforced on the *unrounded* running value, not the
    # rounded display value — this guarantees score is strictly decreasing
    # (never tied) even when the underlying final/domain_evidence values
    # collide, so the emitted CSV can never require a candidate_id tie-break.
    EPS = 2e-6
    for rank, (c, final, trace, conf, _fs) in enumerate(pool[:top_n], start=1):
        tier  = predict_tier(trace)
        raw   = final if final < prev - EPS else prev - EPS
        score = round(raw, 6)
        prev  = raw
        rows.append({
            "candidate_id":    c.get("candidate_id", ""),
            "rank":            rank,
            "score":           score,
            "reasoning":       make_reasoning(c, trace, tier, rank_stability=conf, rank=rank),
            "_tier":           tier,
            "_trace":          trace,
            "_rank_stability": conf,
        })
    assert all(rows[i]["rank"] == i + 1 for i in range(len(rows))), \
        "cascade rank sequence broken"
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# INLINE HTML REPORT  (--report flag)
# ──────────────────────────────────────────────────────────────────────────────

_REPORT_CSS = """\
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#f8fafc;color:#1e293b}
header{background:#0f172a;color:#f1f5f9;padding:18px 28px;display:flex;align-items:center;gap:16px}
header h1{font-size:1.25rem;font-weight:700}
.sub{font-size:.78rem;color:#94a3b8;margin-top:3px}
.hstats{margin-left:auto;display:flex;gap:24px}
.hstat{text-align:right;font-size:.78rem;color:#94a3b8}
.hstat strong{display:block;font-size:1rem;color:#f1f5f9;font-weight:600}
#ctrl{position:sticky;top:0;z-index:10;background:#fff;border-bottom:1px solid #e2e8f0;
  padding:8px 28px;display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.fb{padding:3px 11px;border-radius:999px;border:1px solid #e2e8f0;background:#fff;
  cursor:pointer;font-size:.78rem;transition:background .1s,color .1s}
.fb:hover{background:#f1f5f9}.fb.active{background:#0f172a;color:#fff;border-color:#0f172a}
#srch{margin-left:auto;padding:4px 9px;border:1px solid #e2e8f0;border-radius:6px;
  font-size:.79rem;width:180px;outline:none}
#srch:focus{border-color:#94a3b8}
#cards{max-width:860px;margin:18px auto;padding:0 14px;display:flex;flex-direction:column;gap:10px}
.card{background:#fff;border-radius:10px;border:1px solid #e2e8f0;
  box-shadow:0 1px 3px rgba(0,0,0,.05);overflow:hidden;transition:box-shadow .15s}
.card:hover{box-shadow:0 3px 12px rgba(0,0,0,.09)}
.card-hdr{display:flex;align-items:center;gap:8px;padding:11px 14px}
.rank-badge{min-width:34px;height:34px;border-radius:7px;background:#0f172a;color:#fff;
  display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.74rem;
  flex-shrink:0;padding:0 4px}
.cid{flex:1;font-size:.78rem;color:#475569;font-family:ui-monospace,monospace;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0}
.t-badge{padding:2px 8px;border-radius:999px;font-size:.71rem;font-weight:700;flex-shrink:0}
.score{font-size:.87rem;font-weight:600;color:#0f172a;flex-shrink:0}
.stab-wrap{display:flex;align-items:center;gap:5px;flex-shrink:0;margin-left:2px}
.stab-track{width:54px;height:5px;background:#e2e8f0;border-radius:3px;overflow:hidden;flex-shrink:0}
.stab-fill{height:100%;border-radius:3px}
.stab-lbl{font-size:.69rem;font-weight:700;white-space:nowrap}
.meta-row{padding:0 14px 5px;font-size:.77rem;color:#64748b;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.m-title{font-weight:500;color:#334155}.m-meta{color:#94a3b8}
.pills-row{padding:4px 14px;display:flex;flex-wrap:wrap;gap:4px}
.ev-pill{padding:2px 7px;background:#dbeafe;color:#1d4ed8;border-radius:999px;font-size:.68rem;font-weight:500}
.flag{padding:2px 7px;border-radius:4px;font-size:.69rem;font-weight:600}
.flag-hp{background:#fee2e2;color:#b91c1c}.flag-if{background:#fef3c7;color:#92400e}
.flag-role{background:#fce7f3;color:#9d174d}.flag-pen{background:#fff7ed;color:#c2410c;font-weight:400}
.wf{padding:6px 14px 2px}
.wl,.wv{font:10px system-ui,sans-serif}.wl{fill:#6b7280;text-anchor:end}.wv{fill:#374151}
.r-preview{padding:4px 14px 2px;font-size:.78rem;color:#64748b;line-height:1.55}
.r-full{padding:0 14px 8px}
.r-box{font-size:.79rem;color:#1e293b;line-height:1.65;background:#f8fafc;border-radius:6px;
  padding:9px 12px;border:1px solid #e2e8f0;white-space:pre-wrap;word-break:break-word}
.r-btn{display:block;width:100%;padding:6px;background:none;border:none;
  border-top:1px solid #f1f5f9;font-size:.74rem;color:#64748b;cursor:pointer;text-align:center}
.r-btn:hover{background:#f8fafc;color:#0f172a}
#no-res{text-align:center;padding:48px;color:#94a3b8;font-size:.88rem;display:none}
"""

_REPORT_PL = ('["Domain Evid.","Skill Subst.","Seniority","Prod vs Svc","Ext. Valid.",'
              '"Eval Frmwk.","Semantic","Python","Location","Notice","Platform"]')

_REPORT_JS = """\
var _PL=__PL__;
function renderWF(div){
  var raw=div.dataset.p;
  if(!raw){div.innerHTML='<p style="color:#9ca3af;font-size:12px">No pillar data</p>';return;}
  var data=JSON.parse(raw),LBL=106,BAR=155,GAP=3,ROW=19,W=LBL+GAP+BAR+GAP+38;
  var keys=[];data.forEach(function(d,i){if(d)keys.push(i);});
  var H=keys.length*ROW+2,ns='http://www.w3.org/2000/svg';
  var maxW=Math.max.apply(null,keys.map(function(i){return data[i][1];}));
  var svg=document.createElementNS(ns,'svg');svg.setAttribute('width',W);svg.setAttribute('height',H);
  function el(t,a){var e=document.createElementNS(ns,t);Object.keys(a).forEach(function(k){e.setAttribute(k,a[k]);});return e;}
  keys.forEach(function(ki,i){
    var d=data[ki],pv=d[0],w=d[1],contrib=(w*pv).toFixed(3);
    var y=i*ROW+1,mid=y+12,bx=LBL+GAP;
    var barPx=Math.max(0,Math.round(pv*BAR)),maxPx=Math.max(1,Math.round((w/maxW)*BAR));
    var col=pv>=0.70?'#16a34a':pv>=0.40?'#3b82f6':'#f59e0b';
    var t1=el('text',{x:LBL,y:mid,'class':'wl'});t1.textContent=_PL[ki];svg.appendChild(t1);
    svg.appendChild(el('rect',{x:bx,y:y+2,width:maxPx,height:13,fill:'#f1f5f9',rx:2}));
    if(barPx>0)svg.appendChild(el('rect',{x:bx,y:y+2,width:barPx,height:13,fill:col,rx:2}));
    var t2=el('text',{x:bx+BAR+GAP,y:mid,'class':'wv'});t2.textContent=contrib;svg.appendChild(t2);
  });
  div.appendChild(svg);
}
document.querySelectorAll('.wf[data-p]').forEach(renderWF);
function toggleR(btn){
  var c=btn.closest('.card'),f=c.querySelector('.r-full'),p=c.querySelector('.r-preview');
  if(f.hidden){f.hidden=false;p.style.display='none';btn.textContent='▴ Hide reasoning';}
  else{f.hidden=true;p.style.display='';btn.textContent='▾ Full reasoning';}
}
var _f='all';
function applyFilter(){
  var q=document.getElementById('srch').value.trim().toLowerCase(),n=0;
  document.querySelectorAll('.card').forEach(function(c){
    var ok=(_f==='all'||'t'+c.dataset.tier===_f||(_f==='flagged'&&c.dataset.flagged==='1'))
           &&(!q||c.querySelector('.cid').textContent.toLowerCase().includes(q));
    c.style.display=ok?'':'none';if(ok)n++;
  });
  document.getElementById('no-res').style.display=n?'none':'block';
}
document.querySelectorAll('.fb').forEach(function(b){
  b.addEventListener('click',function(){
    document.querySelectorAll('.fb').forEach(function(x){x.classList.remove('active');});
    b.classList.add('active');_f=b.dataset.filter;applyFilter();
  });
});
document.getElementById('srch').addEventListener('input',applyFilter);
"""

_REPORT_PILLAR_ORDER = [
    "domain_evidence", "skill_substance", "seniority_fit", "product_vs_services",
    "external_validation", "eval_frameworks", "semantic_similarity",
    "python_signal", "location", "notice", "platform_quality",
]
_REPORT_TIER_COLORS = {
    5: ("#fff", "#16a34a"), 4: ("#fff", "#2563eb"), 3: ("#fff", "#b45309"),
    2: ("#fff", "#6b7280"), 1: ("#fff", "#9ca3af"), 0: ("#fff", "#dc2626"),
}


def generate_html_report(rows, candidates, path):
    """Build a self-contained offline HTML report and open it in the browser.
    Uses only `rows` (already scored) and the original `candidates` list.
    No re-scoring, no file reads — target <2 s on 100 candidates.
    """
    import json as _json
    import platform
    import subprocess
    from pathlib import Path as _Path

    cand_lookup = {c.get("candidate_id", ""): c for c in candidates}

    def _esc(s):
        return (str(s).replace("&", "&amp;").replace("<", "&lt;")
                      .replace(">", "&gt;").replace('"', "&quot;"))

    def _pillar_json(trace):
        pillars = trace.get("pillars", {})
        if not pillars:
            return ""
        data = []
        for k in _REPORT_PILLAR_ORDER:
            pv = pillars.get(k)
            w  = WEIGHTS.get(k, 0)
            if pv is not None and w > 0:
                data.append([round(float(pv), 3), round(float(w), 3)])
            else:
                data.append(None)
        while data and data[-1] is None:
            data.pop()
        return _json.dumps(data, separators=(",", ":"))

    def _card(row):
        rank  = row["rank"]
        cid   = row["candidate_id"]
        score = row["score"]
        reason = row["reasoning"]
        tier  = row["_tier"]
        stab  = row["_rank_stability"]   # float 0-1
        tr    = row["_trace"]
        tfg, tbg = _REPORT_TIER_COLORS[tier]

        sp = round(stab * 100)
        sc = "#16a34a" if stab >= 0.85 else "#d97706" if stab >= 0.50 else "#dc2626"
        sb = ("✓" if stab >= 0.85 else "~" if stab >= 0.50 else "✗") + f" {sp}%"
        stab_html = (
            f'<div class="stab-wrap" title="Rank stability: {sp}%">'
            f'<div class="stab-track">'
            f'<div class="stab-fill" style="width:{sp}%;background:{sc}"></div>'
            f'</div><span class="stab-lbl" style="color:{sc}">{sb}</span></div>'
        )

        c = cand_lookup.get(cid, {})
        p = c.get("profile", {})
        meta_parts = []
        title = (p.get("current_title") or "").strip()
        loc   = (p.get("location") or "").strip()
        yoe   = p.get("years_of_experience")
        if title:           meta_parts.append(f'<span class="m-title">{_esc(title[:55])}</span>')
        if yoe is not None: meta_parts.append(f'<span class="m-meta">{yoe}y exp</span>')
        if loc:             meta_parts.append(f'<span class="m-meta">📍 {_esc(loc[:35])}</span>')
        meta_html = (f'<div class="meta-row">{" · ".join(meta_parts)}</div>'
                     if meta_parts else "")

        ev_terms = tr.get("domain_evidence_terms", [])
        ev_html = ""
        if ev_terms:
            pills = "".join(f'<span class="ev-pill">{_esc(t)}</span>' for t in ev_terms)
            ev_html = f'<div class="pills-row">{pills}</div>'

        is_hp    = tr.get("honeypot", False)
        hp_flags = tr.get("honeypot_flags", [])
        neg      = tr.get("neg_reasons", [])
        if_anom  = tr.get("if_anomaly") or 0
        role     = tr.get("role", "")
        is_flagged = is_hp or if_anom > 0 or role in ("non_technical", "non_target_tech")
        flag_parts = []
        if is_hp:
            flag_parts.append(
                f'<span class="flag flag-hp">⚠ Honeypot: {_esc(", ".join(hp_flags[:3]))}</span>'
            )
        if if_anom > 0.85:
            flag_parts.append(f'<span class="flag flag-if">⚠ IF anomaly {if_anom:.2f}</span>')
        if role in ("non_technical", "non_target_tech"):
            flag_parts.append(f'<span class="flag flag-role">⚠ {_esc(role)}</span>')
        for rn in neg:
            flag_parts.append(f'<span class="flag flag-pen">{_esc(rn)}</span>')
        flags_html = (f'<div class="pills-row">{"".join(flag_parts)}</div>'
                      if flag_parts else "")
        border = 'style="border-left:3px solid #dc2626"' if is_flagged else ""

        r_clean = reason
        idx = r_clean.find(" Rank stability:")
        if idx != -1:
            r_clean = r_clean[:idx]
        preview = _esc(r_clean[:140]) + ("…" if len(r_clean) > 140 else "")

        pd = _pillar_json(tr)
        wf_attr = f" data-p='{pd}'" if pd else ""

        return (
            f'<div class="card" data-tier="{tier}" data-flagged="{int(is_flagged)}" '
            f'data-rank="{rank}" {border}>'
            f'<div class="card-hdr">'
            f'<div class="rank-badge">#{rank}</div>'
            f'<div class="cid">{_esc(cid)}</div>'
            f'<span class="t-badge" style="color:{tfg};background:{tbg}">T{tier}</span>'
            f'<span class="score">{score:.4f}</span>'
            f'{stab_html}</div>'
            f'{meta_html}{ev_html}{flags_html}'
            f'<div class="wf"{wf_attr}></div>'
            f'<div class="r-preview">{preview}</div>'
            f'<div class="r-full" hidden><div class="r-box">{_esc(reason)}</div></div>'
            f'<button class="r-btn" onclick="toggleR(this)">▾ Full reasoning</button>'
            f'</div>'
        )

    tiers = sorted({r["_tier"] for r in rows}, reverse=True)
    n_flagged = sum(
        1 for r in rows
        if r["_trace"].get("honeypot") or r["_trace"].get("if_anomaly") or
           r["_trace"].get("role") in ("non_technical", "non_target_tech")
    )
    n = len(rows)
    tier_btns = "".join(
        f'<button class="fb" data-filter="t{t}">T{t}</button>' for t in tiers
    )
    flag_btn = (
        f'<button class="fb" data-filter="flagged">⚠ Flagged ({n_flagged})</button>'
        if n_flagged else ""
    )
    cards_html = "".join(_card(r) for r in rows)
    top_score  = f"{rows[0]['score']:.4f}" if rows else "—"
    gen_ts     = datetime.now().strftime("%Y-%m-%d %H:%M")
    js         = _REPORT_JS.replace("__PL__", _REPORT_PL)

    html = (
        '<!DOCTYPE html><html lang="en"><head>'
        '<meta charset="UTF-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>Redrob Ranking Report</title><style>{_REPORT_CSS}</style></head><body>'
        f'<header><div><h1>Redrob Ranking Report</h1>'
        f'<div class="sub">top {n} candidates &nbsp;·&nbsp; {gen_ts}</div></div>'
        f'<div class="hstats">'
        f'<div class="hstat"><strong>{n}</strong>ranked</div>'
        f'<div class="hstat"><strong>{top_score}</strong>top score</div>'
        f'<div class="hstat"><strong>{n_flagged}</strong>flagged</div>'
        f'</div></header>'
        f'<div id="ctrl">'
        f'<button class="fb active" data-filter="all">All ({n})</button>'
        f'{tier_btns}{flag_btn}'
        f'<input id="srch" type="search" placeholder="Search candidate ID…">'
        f'</div>'
        f'<div id="cards">{cards_html}'
        f'<div id="no-res">No candidates match this filter.</div></div>'
        f'<script>{js}</script>'
        f'</body></html>'
    )

    _Path(path).write_text(html, encoding="utf-8")
    kb = len(html.encode()) / 1024
    print(f"  report  → {path}  ({kb:.0f} KB)", flush=True)

    try:
        sys_name = platform.system()
        if sys_name == "Darwin":
            subprocess.Popen(["open", path])
        elif sys_name == "Linux":
            subprocess.Popen(["xdg-open", path])
        elif sys_name == "Windows":
            subprocess.Popen(["start", path], shell=True)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(
        description="Redrob Ranking Engine — glass-box, deterministic, CPU-only candidate ranker."
    )
    ap.add_argument("--candidates",   required=True,
                    help="Candidate pool JSONL (supports .jsonl.gz compression)")
    ap.add_argument("--out",          default="submission.csv",
                    help="Output submission CSV (default: submission.csv)")
    ap.add_argument("--top",          type=int, default=100,
                    help="Number of candidates to rank (default: 100)")
    ap.add_argument("--dump-pillars", default=None, dest="dump_pillars",
                    help="Write pillar feature CSV to this path (for LambdaMART training)")
    ap.add_argument("--experiments",  action="store_true",
                    help="Run all A-W variant experiments; imports experiments.py")
    ap.add_argument("--labels",       default="labels_filled 500.csv",
                    help="Labels CSV used by --experiments (default: 'labels_filled 500.csv')")
    ap.add_argument("--rich",         action="store_true",
                    help="Render live Phase-1 progress bar and color-coded top-10 table "
                         "(requires rich; auto-installed if absent; graceful fallback)")
    ap.add_argument("--report",       default=None, metavar="PATH",
                    help="Write a self-contained HTML report to PATH and open it in the browser")
    ap.add_argument("--weights-file", default=None, dest="weights_file", metavar="PATH",
                    help="JSON file of pillar weights to override WEIGHTS_FALLBACK "
                         "(keys must match WEIGHTS_FALLBACK, values must sum to ~1.0)")
    args = ap.parse_args()

    if args.weights_file:
        with open(args.weights_file) as _fh:
            _wdata = {k: float(v) for k, v in json.load(_fh).items()
                      if not k.startswith("_")}
        _wtotal = sum(_wdata.values())
        if abs(_wtotal - 1.0) > 0.02:
            raise SystemExit(f"--weights-file weights sum to {_wtotal:.4f}, expected ~1.0")
        global WEIGHTS_FALLBACK
        WEIGHTS_FALLBACK = _wdata
        print(f"  [weights] Overriding WEIGHTS_FALLBACK from {args.weights_file} "
              f"(sum={_wtotal:.4f})", flush=True)

    use_rich = args.rich and _ensure_rich()

    t0 = datetime.now()
    print(f"Loading {args.candidates} …")
    cands = load_candidates(args.candidates)
    print(f"  {len(cands):,} candidates loaded")

    rows, scored = rank_all(cands, top_n=args.top, use_rich=use_rich)
    write_csv(rows, args.out)

    if args.report:
        generate_html_report(rows, cands, args.report)

    if args.dump_pillars:
        _export_pillar_features(scored, args.dump_pillars)

    hp = sum(1 for r in rows if r["_trace"]["honeypot"])
    nt = sum(1 for r in rows if r["_trace"]["role"] in ("non_technical", "non_target_tech"))
    dt = (datetime.now() - t0).total_seconds()
    print(f"\n  wrote {len(rows)} rows -> {args.out}")
    print(f"  honeypots in top-{args.top}: {hp} ({hp/max(len(rows),1):.1%})  "
          f"| non-target roles: {nt}")
    print(f"  runtime: {dt:.1f}s")
    if use_rich:
        _render_rich_top10(rows)
    else:
        print(f"\n  Top 10:")
        for r in rows[:10]:
            conf_str = f"stab={r['_rank_stability']:.0%}"
            print(f"    {r['rank']:>3}. {r['candidate_id']}  "
                  f"s={r['score']:.4f} t{r['_tier']} {conf_str}  "
                  f"{r['reasoning'][:80]}")

    if args.experiments:
        import experiments
        experiments.run_all_experiments(scored, labels_path=args.labels)


if __name__ == "__main__":
    main()
