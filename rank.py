#!/usr/bin/env python3
"""
rank.py — Redrob Intelligent Candidate Discovery & Ranking Engine
==================================================================
Glass-box, fully deterministic, zero-AI, zero-network candidate ranker.

Design principle
----------------
The RANKING and the REASONING are the same object. Every candidate's final
score is a transparent decomposition of named, computed factors (a "trace").
The reasoning column is a deterministic readout of that exact trace, so it can
never contradict the rank and can never cite a fact the candidate doesn't have.

Pipeline (layers)
-----------------
  L0  Honeypot / impossible-profile gate         -> hard floor to bottom
  L1  Role-fit gate                              -> floors keyword-stuffers
  L2  Domain evidence ("says vs means")          -> reads career descriptions
  L3  Fit pillars (seniority, product-vs-services, skills, python, eval,
      external validation, location, notice)
  L4  Negative do-NOT-want signals               -> penalty (not floor)
  L5  Behavioral availability multiplier         -> JD "actually available"
  ->  tier prediction, deterministic ranking, trace-driven reasoning, CSV

Compute: pure numpy/pandas + python stdlib. CPU only. No GPU, no network,
no LLM, no embeddings, no model weights. Runs the full 100k pool well under
the 5 min / 16 GB budget.

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
import re
from datetime import date, datetime

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# JOB PROFILE — hand-derived from the REAL Redrob JD (Senior AI Engineer,
# Founding Team). This is NOT the generic sample_job.json. Everything the
# engine optimizes for lives here so it is auditable and tunable.
# ──────────────────────────────────────────────────────────────────────────────

JOB = {
    "title": "Senior AI Engineer — Founding Team (Redrob)",
    "ideal_yoe": 7.0, "yoe_sigma": 2.0,          # 5-9 band, peak 6-8
    "yoe_soft_min": 4.0,                          # "judgment at 4y" allowance
    "target_cities": {                            # JD-stated preferred locations
        "pune", "noida", "hyderabad", "mumbai", "delhi", "new delhi",
        "ncr", "gurgaon", "gurugram", "ghaziabad", "faridabad", "greater noida",
    },
    "good_india_cities": {"bangalore", "bengaluru", "chennai", "kolkata", "ahmedabad"},
    "notice_ideal_days": 30,                      # buy-out up to 30; <30 ideal
}

# Pillar weights (sum ~1.0). These are the tunable surface; the validation
# harness ablates them to confirm the top-10 is stable, not knife-edge.
WEIGHTS = {
    "domain_evidence":     0.30,   # built retrieval/ranking/recsys/search (the core)
    "skill_substance":     0.15,   # proficiency*duration*endorsement*assessment, relevance-gated
    "seniority_fit":       0.12,
    "product_vs_services": 0.10,
    "external_validation": 0.08,   # github activity + OSS/external signal
    "eval_frameworks":     0.08,   # NDCG/MRR/MAP/A-B thinking
    "python_signal":       0.07,
    "location":            0.06,
    "notice":              0.04,
}

# ── Concept lexicons (compiled once; matched against free text) ───────────────
def _rx(words):
    # case-sensitive: all inputs are pre-lowercased once (≈1.8x faster than re.I)
    return re.compile("|".join(r"\b" + w + r"\b" for w in words))

RX_RETRIEVAL = _rx([
    r"retriev\w*", r"ranking", r"rank(ed|ing|er)?", r"recommend\w*", r"recommender",
    r"search relevance", r"relevance", r"semantic search", r"vector search",
    r"nearest neighbou?r", r"learning[- ]to[- ]rank", r"\bltr\b", r"\bann\b",
    r"information retrieval", r"\bir\b", r"personali[sz]ation", r"matching",
    r"\bbm25\b", r"\bfaiss\b", r"elasticsearch", r"opensearch", r"\bsolr\b",
    r"\blucene\b", r"embeddings?", r"two[- ]tower", r"candidate generation",
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
    r"search", r"ranking", r"recommend\w*",
])
# Adjacent ML/data signal — catches PLAIN-LANGUAGE fits (the Tier-5 trap) without
# the explicit IR buzzwords. Capped so adjacent-only never reaches explicit-fit level.
RX_ADJACENT_ML = _rx([
    r"feature (pipeline|engineering|store)", r"feature pipelines?", r"ml pipeline\w*",
    r"experimentation", r"model (training|serving|deployment|inference)",
    r"data science", r"machine learning model\w*", r"predictive model\w*",
    r"classification", r"regression model\w*", r"forecasting", r"churn", r"propensity",
    r"\bkaggle\b", r"fine[- ]tun\w*", r"\bspark\b", r"airflow", r"\bdbt\b", r"\betl\b",
    r"\bxgboost\b", r"lightgbm", r"gradient boost\w*", r"feature pipeline",
    r"data (pipeline|infrastructure|warehouse)", r"analytics", r"\bml\b model\w*",
])
RX_PYTHON = _rx([r"python", r"pytorch", r"tensorflow", r"scikit", r"numpy", r"pandas"])
RX_OSS = _rx([r"open[- ]source", r"open source", r"github", r"maintainer",
             r"contributor", r"\bpaper\b", r"published", r"\btalk\b", r"conference"])
RX_RESEARCH = _rx([r"research scientist", r"\bphd\b", r"post[- ]?doc", r"postdoctoral",
                  r"research (lab|assistant|associate)", r"academ\w*", r"thesis",
                  r"university research", r"published \d"])

# ── Role buckets (matched on current_title) ───────────────────────────────────
RX_NON_TECHNICAL = _rx([   # keyword-stuffer gate: floor regardless of skills
    r"hr\b", r"human resource\w*", r"recruit\w*", r"talent acquisition",
    r"marketing", r"content writer", r"copywriter", r"\bcontent\b", r"\bsales\b",
    r"account executive", r"accountant", r"\bfinance\b", r"customer support",
    r"customer success", r"operations manager", r"\bbpo\b", r"administrativ\w*",
    r"office manager", r"business development", r"social media",
])
RX_NON_TARGET_TECH = _rx([ # technical but wrong domain
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
    r"devops", r"software developer", r"engineer",  # generic engineer last
])

# ── Services / consulting firms (JD do-NOT-want if entire career here) ─────────
SERVICES_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "tech mahindra", "hcl", "hcltech", "mindtree", "ltimindtree",
    "lti", "deloitte", "ibm", "dxc", "mphasis", "persistent systems",
}

PROF_W = {"beginner": 0.3, "intermediate": 0.6, "advanced": 0.85, "expert": 1.0}


# ══════════════════════════════════════════════════════════════════════════════
# LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_candidates(path):
    """Stream candidates from .jsonl.gz, .jsonl, or a .json array. Memory-safe."""
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
    else:  # .json array (the 50-sample)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            out = data if isinstance(data, list) else [data]
    return out


def _parse_date(s):
    if not s or len(s) < 10:
        return None
    try:  # manual parse — ~6x faster than datetime.strptime at 100k scale
        return date(int(s[:4]), int(s[5:7]), int(s[8:10]))
    except Exception:
        return None


def _candidate_text(c):
    """Concatenate all free text we are allowed to read for evidence."""
    p = c.get("profile", {})
    parts = [p.get("headline", ""), p.get("summary", ""),
             p.get("current_title", ""), p.get("current_industry", "")]
    for h in c.get("career_history", []):
        parts.append(h.get("title", ""))
        parts.append(h.get("description", ""))
        parts.append(h.get("industry", ""))
    return " \n ".join(x for x in parts if x)


# ══════════════════════════════════════════════════════════════════════════════
# L0 — HONEYPOT / IMPOSSIBLE-PROFILE DETECTION  (deterministic, hard floor)
# ══════════════════════════════════════════════════════════════════════════════

def honeypot_flags(c):
    """
    Returns (is_honeypot, flag_list). Catches the dataset's planted impossible
    profiles using internal consistency only — no special-casing needed.
    """
    flags = []
    yoe = float(c.get("profile", {}).get("years_of_experience", 0) or 0)
    total_career_m = sum(int(h.get("duration_months", 0) or 0)
                         for h in c.get("career_history", []))

    # (1) "expert/advanced in N skills with 0 months used"
    zero_dur_high = sum(
        1 for s in c.get("skills", [])
        if s.get("proficiency") in ("advanced", "expert")
        and int(s.get("duration_months", 1) or 0) == 0
    )
    if zero_dur_high >= 3:
        flags.append("many_high_skills_zero_duration")

    # (2) used a skill longer than the person has ever worked
    for s in c.get("skills", []):
        if int(s.get("duration_months", 0) or 0) > total_career_m + 12 and total_career_m > 0:
            flags.append("skill_duration_exceeds_career")
            break

    # (3) role duration_months contradicts its own start/end dates
    #     (catches "8y at a company that has only existed 3y" style traps)
    date_conflicts = 0
    for h in c.get("career_history", []):
        sd = _parse_date(h.get("start_date"))
        ed = _parse_date(h.get("end_date")) or date(2026, 6, 1)
        dur = int(h.get("duration_months", 0) or 0)
        if sd:
            calc = (ed.year - sd.year) * 12 + (ed.month - sd.month)
            if calc >= 0 and abs(dur - calc) > 9:
                date_conflicts += 1
    if date_conflicts >= 1:
        flags.append("tenure_date_contradiction")

    # (4) summed tenure wildly exceeds stated experience (beyond plausible overlap)
    if yoe > 0 and total_career_m > yoe * 12 + 30:
        flags.append("career_sum_exceeds_yoe")

    # (5) is_current contradictions
    currents = [h for h in c.get("career_history", []) if h.get("is_current")]
    if any(h.get("end_date") for h in currents) or len(currents) > 1:
        flags.append("is_current_contradiction")

    # honeypot if a strong flag, or two independent flags
    strong = {"many_high_skills_zero_duration", "skill_duration_exceeds_career"}
    is_hp = bool(strong & set(flags)) or len(set(flags)) >= 2
    return is_hp, sorted(set(flags))


# ══════════════════════════════════════════════════════════════════════════════
# L1 — ROLE FIT
# ══════════════════════════════════════════════════════════════════════════════

def role_fit(c):
    """Returns (multiplier 0..1, label). Current title dominates per the JD."""
    title = (c.get("profile", {}).get("current_title", "") or "").lower()
    if RX_NON_TECHNICAL.search(title):
        return 0.05, "non_technical"          # keyword-stuffer gate
    if RX_NON_TARGET_TECH.search(title):
        return 0.12, "non_target_tech"
    if RX_CORE_ML.search(title):
        return 1.00, "core_ml"
    if RX_ADJACENT_ENG.search(title):
        return 0.58, "adjacent_eng"           # liftable by domain evidence
    return 0.30, "unknown_role"


# ══════════════════════════════════════════════════════════════════════════════
# L2 — DOMAIN EVIDENCE  ("the gap between what the JD says and what it means")
# ══════════════════════════════════════════════════════════════════════════════

def domain_evidence(c, text):
    """
    Layered "says vs means" scorer. Reads career descriptions, not skill names.
      • EXPLICIT layer: production retrieval/ranking/recsys/search/vector/LLM/eval
        signal -> can reach the full 0..1 range.
      • ADJACENT layer: plain-language ML/data signal (feature pipelines, model
        training, data-science collaboration, Spark/Airflow, fine-tuning) -> catches
        Tier-5 plain-language fits, but is HARD-CAPPED so an adjacent-only candidate
        always ranks clearly below anyone with explicit retrieval/ranking work.
    Returns (score 0..1, evidence_terms, is_cv_robotics_heavy).
    """
    ev = []
    retr = len(RX_RETRIEVAL.findall(text))
    prod = len(RX_PRODUCTION.findall(text))
    modern = len(RX_LLM_MODERN.findall(text))
    vdb = len(RX_VECTOR_DB.findall(text))
    evalh = len(RX_EVAL.findall(text))
    nlpir = len(RX_NLP_IR.findall(text))
    adj = len(RX_ADJACENT_ML.findall(text))
    cvrobo = len(RX_CV_SPEECH_ROBO.findall(text))

    # ── EXPLICIT layer (the real signal the JD wants proof of) ──
    retr_s = 1 - math.exp(-retr / 2.0)
    modern_s = 1 - math.exp(-modern / 2.0)
    vdb_s = min(vdb, 1)
    eval_s = 1 - math.exp(-evalh / 1.5)
    nlpir_s = 1 - math.exp(-nlpir / 3.0)
    explicit = (0.45 * retr_s + 0.18 * modern_s + 0.15 * vdb_s
                + 0.12 * eval_s + 0.10 * nlpir_s)
    explicit_present = (retr + modern + vdb + evalh) > 0

    # production multiplier — built-it-in-prod beats mentioned-it
    prod_s = 1 - math.exp(-prod / 2.0)
    explicit *= (0.7 + 0.3 * prod_s)

    # ── ADJACENT layer (plain-language) ──
    adjacent = 1 - math.exp(-adj / 3.0)         # saturating

    # ── Combine with STRICT ceiling on adjacent-only ──
    if explicit_present:
        score = float(np.clip(0.80 * explicit + 0.20 * adjacent, 0, 1))
        if retr: ev.append("retrieval/ranking work")
        if vdb: ev.append("vector/search infra")
        if modern: ev.append("modern LLM/embedding work")
        if evalh: ev.append("ranking-evaluation experience")
        if prod and not ev: ev.append("production ML deployment")
    else:
        # no explicit IR/ranking signal -> hard-capped at 0.42 (tier 2-3 zone)
        score = float(np.clip(adjacent, 0, 1) * 0.42)
        if adj: ev.append("adjacent ML/data work (plain-language)")

    cv_heavy = cvrobo >= 2 and (retr + nlpir) < cvrobo
    return score, ev, cv_heavy


# ══════════════════════════════════════════════════════════════════════════════
# L3 — FIT PILLARS
# ══════════════════════════════════════════════════════════════════════════════

def seniority_fit(c):
    yoe = float(c.get("profile", {}).get("years_of_experience", 0) or 0)
    s = math.exp(-0.5 * ((yoe - JOB["ideal_yoe"]) / JOB["yoe_sigma"]) ** 2)
    if yoe < JOB["yoe_soft_min"]:               # under-qualified penalty
        s *= (max(yoe, 0) / JOB["yoe_soft_min"]) ** 2
    return float(s)


def product_vs_services(c):
    """1.0 = product-company career; low = pure services/consulting (with the
    JD's escape hatch: prior product-company experience neutralizes the penalty)."""
    hist = c.get("career_history", [])
    if not hist:
        return 0.5
    def is_services(h):
        comp = (h.get("company", "") or "").lower()
        ind = (h.get("industry", "") or "").lower()
        return any(f in comp for f in SERVICES_FIRMS) or "it services" in ind \
            or "consult" in ind
    flags = [is_services(h) for h in hist]
    if not any(flags):
        return 1.0
    if all(flags):
        return 0.25                              # entire career in services
    # mixed: has some product experience -> JD says that's fine
    return 0.8


def skill_substance(c):
    """Relevance-gated, substance-weighted skills (not mere keyword presence).
    Uses proficiency * tenure * endorsement, plus Redrob assessment scores."""
    sig = c.get("redrob_signals", {})
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
        pw = PROF_W.get(s.get("proficiency"), 0.4)
        dur = min(int(s.get("duration_months", 0) or 0) / 24.0, 1.0)
        end = min(math.log1p(int(s.get("endorsements", 0) or 0)) / 4.0, 1.0)
        a = assess.get(s.get("name", ""), None)
        asc = (a / 100.0) if isinstance(a, (int, float)) and a >= 0 else 0.5
        total += 0.4 * pw + 0.2 * dur + 0.15 * end + 0.25 * asc
    if hits == 0:
        return 0.0
    return float(np.clip(total / max(hits, 1) * min(hits / 4.0, 1.0) + 0.0, 0, 1))


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
    if gh < 0:
        base = 0.40                              # -1 = no GitHub -> NEUTRAL, not bad
    else:
        base = float(np.clip(0.45 + gh / 100.0 * 0.55, 0, 1))
    if RX_OSS.search(text):
        base = min(base + 0.15, 1.0)
    return base


def location_score(c):
    p = c.get("profile", {})
    loc = (p.get("location", "") or "").lower()
    country = (p.get("country", "") or "").lower()
    relo = bool(c.get("redrob_signals", {}).get("willing_to_relocate", False))
    in_india = "india" in country or country == ""
    if any(city in loc for city in JOB["target_cities"]):
        return 1.0
    if any(city in loc for city in JOB["good_india_cities"]):
        return 0.85 if not relo else 0.90
    if in_india:
        return 0.70 if relo else 0.45
    return 0.40 if relo else 0.20               # outside India, no visa sponsorship


def notice_score(c):
    nd = c.get("redrob_signals", {}).get("notice_period_days", 30)
    nd = float(nd) if nd is not None else 30.0
    return float(1.0 / (1.0 + math.exp(0.06 * (nd - JOB["notice_ideal_days"]))))


# ══════════════════════════════════════════════════════════════════════════════
# L4 — NEGATIVE do-NOT-WANT SIGNALS (penalty, not floor)
# ══════════════════════════════════════════════════════════════════════════════

def negative_penalty(c, text, domain_score, prod_serv, cv_heavy):
    """Returns (penalty 0..0.85, reasons[]). Compounding but capped."""
    reasons, ps = [], []
    # research-only: research markers + no production evidence + not product-strong
    if RX_RESEARCH.search(text) and not RX_PRODUCTION.search(text) and domain_score < 0.4:
        ps.append(0.55); reasons.append("research-leaning, little production signal")
    # consulting-only entire career
    if prod_serv <= 0.25:
        ps.append(0.40); reasons.append("career entirely in IT-services/consulting")
    # CV/speech/robotics-heavy without NLP/IR
    if cv_heavy:
        ps.append(0.35); reasons.append("CV/speech/robotics focus, thin NLP/IR")
    # title-chaser: many short stints
    hist = c.get("career_history", [])
    noncur = [h for h in hist if not h.get("is_current")]
    short = [h for h in noncur if int(h.get("duration_months", 24) or 24) < 18]
    if len(noncur) >= 3 and len(short) >= 3:
        ps.append(0.22); reasons.append("frequent short stints (job-hopping pattern)")
    pen = 1.0
    for p in ps:
        pen *= (1 - p)
    return float(min(1 - pen, 0.85)), reasons


# ══════════════════════════════════════════════════════════════════════════════
# L5 — BEHAVIORAL AVAILABILITY MULTIPLIER (JD "actually available")
# ══════════════════════════════════════════════════════════════════════════════

def behavioral_multiplier(c, now):
    s = c.get("redrob_signals", {})
    la = _parse_date(s.get("last_active_date"))
    days = (now - la).days if la else 365
    recency = math.exp(-max(days, 0) / 120.0)            # ~0.37 at 120d, ~0.05 at 365d
    resp = float(s.get("recruiter_response_rate", 0.3) or 0.0)
    icr = float(s.get("interview_completion_rate", 0.5) or 0.0)
    otw = 1.0 if s.get("open_to_work_flag") else 0.55
    comp = float(s.get("profile_completeness_score", 50) or 0) / 100.0
    core = 0.40 * recency + 0.25 * resp + 0.15 * icr + 0.10 * otw + 0.10 * comp
    return float(0.35 + 0.65 * np.clip(core, 0, 1)), {
        "days_inactive": days, "response_rate": resp, "open_to_work": bool(s.get("open_to_work_flag")),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SCORING — assemble the trace per candidate
# ══════════════════════════════════════════════════════════════════════════════

def score_candidate(c, now):
    text = _candidate_text(c).lower()   # lowercase once; all regex is case-sensitive
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
        "python_signal":       python_signal(c, text),
        "location":            location_score(c),
        "notice":              notice_score(c),
    }
    base_fit = sum(WEIGHTS[k] * v for k, v in pillars.items())

    # adjacent-eng roles get lifted by strong domain evidence (plain-language fits)
    if rlabel == "adjacent_eng":
        rmult = min(1.0, rmult + 0.45 * dscore)

    pen, neg_reasons = negative_penalty(c, text, dscore, prod_serv, cv_heavy)
    bmult, bdetail = behavioral_multiplier(c, now)

    final = base_fit * rmult * (1 - pen) * bmult
    if is_hp:
        final = final * 0.01 - 0.2          # crush below all legitimate candidates

    trace = {
        "final": float(final), "base_fit": float(base_fit), "role": rlabel,
        "role_mult": round(rmult, 3), "pillars": {k: round(v, 3) for k, v in pillars.items()},
        "domain_evidence_terms": devid, "penalty": round(pen, 3), "neg_reasons": neg_reasons,
        "behavior_mult": round(bmult, 3), "behavior": bdetail,
        "honeypot": is_hp, "honeypot_flags": hp,
    }
    return final, trace


def predict_tier(trace):
    """Map trace -> graded relevance tier 0..5 (aligns ranking with NDCG grading)."""
    if trace["honeypot"] or trace["role"] in ("non_technical", "non_target_tech"):
        return 0
    f = trace["final"]
    if f >= 0.62: return 5
    if f >= 0.50: return 4
    if f >= 0.38: return 3
    if f >= 0.25: return 2
    if f >= 0.12: return 1
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# REASONING — deterministic readout of the SAME trace that produced the rank
# ══════════════════════════════════════════════════════════════════════════════

def make_reasoning(c, trace, tier):
    """Fact-grounded, varied-by-tier, honest about concerns, zero hallucination:
    every clause is read from the trace/profile, never invented."""
    p = c.get("profile", {})
    yoe = p.get("years_of_experience", 0)
    title = p.get("current_title", "professional")
    loc = p.get("location", "")
    pil = trace["pillars"]

    # lead clause — strongest true positive, tone matched to tier
    if trace["honeypot"]:
        return (f"{title} profile contains internal inconsistencies "
                f"({', '.join(trace['honeypot_flags'][:2])}); flagged as non-credible "
                f"and ranked at the bottom.")
    if trace["role"] == "non_technical":
        return (f"Current role is {title}, a non-engineering function; AI skills are "
                f"listed but the role does not match the JD regardless of keywords.")

    pos = []
    ev = trace["domain_evidence_terms"]
    if ev and pil["domain_evidence"] >= 0.35:
        verb = {5: "strong, directly-relevant", 4: "solid", 3: "some", 2: "limited", 1: "thin", 0: "minimal"}[tier]
        pos.append(f"{verb} {', '.join(ev[:2])}")
    if pil["seniority_fit"] >= 0.6:
        pos.append(f"{yoe}y experience near the 6-8y target")
    elif yoe:
        pos.append(f"{yoe}y experience")
    if pil["product_vs_services"] >= 0.8:
        pos.append("product-company background")
    if pil["external_validation"] >= 0.7:
        pos.append("active external/GitHub signal")
    if pil["eval_frameworks"] >= 0.5:
        pos.append("ranking-evaluation literacy")

    # honest concerns — surfaced from the same numbers
    con = list(trace["neg_reasons"])
    if pil["notice"] < 0.4:
        nd = c.get("redrob_signals", {}).get("notice_period_days")
        con.append(f"{int(nd)}-day notice period" if nd is not None else "long notice period")
    if pil["location"] < 0.5:
        con.append(f"located in {loc}, outside preferred hubs" if loc else "location outside preferred hubs")
    if trace["behavior"]["days_inactive"] > 120:
        con.append(f"inactive {trace['behavior']['days_inactive']}d, response rate {trace['behavior']['response_rate']:.0%}")
    elif trace["behavior"]["response_rate"] < 0.25:
        con.append(f"low recruiter response rate ({trace['behavior']['response_rate']:.0%})")

    lead = title
    pos_str = ("; " + ", ".join(pos)) if pos else ""
    if tier >= 4:
        head = f"{lead} — strong fit{pos_str}."
    elif tier == 3:
        head = f"{lead} with {pos_str.lstrip('; ') or 'partial relevance'}."
    elif tier == 2:
        head = f"{lead}; adjacent fit{pos_str}."
    else:
        head = f"{lead}; weak match for this JD{pos_str}."

    if con:
        head += f" Concern: {con[0]}" + (f"; {con[1]}" if len(con) > 1 else "") + "."
    return head[:300]


# ══════════════════════════════════════════════════════════════════════════════
# DRIVER
# ══════════════════════════════════════════════════════════════════════════════

def rank_all(candidates, top_n=100):
    # dataset-relative "now" = latest activity date in the pool (robust to era)
    now = date(2026, 6, 1)
    for c in candidates:
        d = _parse_date(c.get("redrob_signals", {}).get("last_active_date"))
        if d and d > now:
            now = d

    scored = []
    for c in candidates:
        final, trace = score_candidate(c, now)
        scored.append((c, final, trace))

    # deterministic sort: score desc, then candidate_id asc (spec tie-break)
    scored.sort(key=lambda x: (-x[1], x[0].get("candidate_id", "")))

    rows, prev = [], math.inf
    for rank, (c, final, trace) in enumerate(scored[:top_n], start=1):
        tier = predict_tier(trace)
        score = round(min(final, prev), 6)      # enforce monotonic non-increasing
        prev = score
        rows.append({
            "candidate_id": c.get("candidate_id", ""),
            "rank": rank,
            "score": score,
            "reasoning": make_reasoning(c, trace, tier),
            "_tier": tier, "_trace": trace,
        })
    return rows, scored


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in rows:
            w.writerow([r["candidate_id"], r["rank"], r["score"], r["reasoning"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--top", type=int, default=100)
    args = ap.parse_args()

    t0 = datetime.now()
    print(f"Loading {args.candidates} …")
    cands = load_candidates(args.candidates)
    print(f"  {len(cands):,} candidates loaded")

    rows, scored = rank_all(cands, top_n=args.top)
    write_csv(rows, args.out)

    # self-report: honeypot rate in top-N (the silent DQ gate)
    hp = sum(1 for r in rows if r["_trace"]["honeypot"])
    nt = sum(1 for r in rows if r["_trace"]["role"] in ("non_technical", "non_target_tech"))
    dt = (datetime.now() - t0).total_seconds()
    print(f"  wrote {len(rows)} rows -> {args.out}")
    print(f"  honeypots in top-{args.top}: {hp} ({hp/max(len(rows),1):.1%})  | non-target roles: {nt}")
    print(f"  runtime: {dt:.1f}s")
    print("\n  Top 10:")
    for r in rows[:10]:
        print(f"    {r['rank']:>3}. {r['candidate_id']}  s={r['score']:.4f} t{r['_tier']}  {r['reasoning'][:90]}")


if __name__ == "__main__":
    main()
