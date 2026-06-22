#!/usr/bin/env python3
"""
validation_harness.py — proves the ranker works BEFORE you submit (blind scoring).
==================================================================================
Four tools, since the hackathon gives no leaderboard feedback:

  1. NDCG / MAP / P@k scorer against a hand-labeled tier file (your ground truth).
  2. Ablation: perturb each pillar weight ±, confirm the top-10 is STABLE
     (reading real signal, not balanced on a knife-edge).
  3. Dual-scorer agreement: an independent strict rule-based scorer vs. the
     weighted engine; only top-10 both agree on are "high-confidence".
  4. Adversarial self-test: inject synthetic keyword-stuffers + honeypots and
     verify the engine floors them (the silent >10% DQ gate).

Usage
-----
    # 1. make a labeling template from your candidate file:
    python validation_harness.py make-labels --candidates sample_candidates.json --out labels_template.csv
    #    -> open labels_template.csv, fill the 'tier' column (0..5) for each row, save.

    # 2. score the engine's ranking against your filled labels:
    python validation_harness.py score --candidates sample_candidates.json --labels labels_filled.csv

    # 3. ablation stability check:
    python validation_harness.py ablate --candidates sample_candidates.json

    # 4. adversarial self-test:
    python validation_harness.py selftest --candidates sample_candidates.json
"""

import argparse
import copy
import csv
import json
import math
from datetime import date

import numpy as np

import rank as R


# ──────────────────────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────────────────────

def dcg(relevances):
    return sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(ranked_rels, k):
    ideal = sorted(ranked_rels, reverse=True)
    idcg = dcg(ideal[:k])
    return dcg(ranked_rels[:k]) / idcg if idcg > 0 else 0.0


def average_precision(ranked_rels, rel_threshold=3):
    hits, score = 0, 0.0
    for i, rel in enumerate(ranked_rels):
        if rel >= rel_threshold:
            hits += 1
            score += hits / (i + 1)
    total_rel = sum(1 for r in ranked_rels if r >= rel_threshold)
    return score / total_rel if total_rel else 0.0


def precision_at_k(ranked_rels, k, rel_threshold=3):
    top = ranked_rels[:k]
    return sum(1 for r in top if r >= rel_threshold) / max(len(top), 1)


# ──────────────────────────────────────────────────────────────────────────────
# 1. LABELING TEMPLATE
# ──────────────────────────────────────────────────────────────────────────────

def make_labels(candidates, out):
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "current_title", "yoe", "headline", "tier"])
        for c in candidates:
            p = c.get("profile", {})
            w.writerow([c.get("candidate_id", ""), p.get("current_title", ""),
                        p.get("years_of_experience", ""), (p.get("headline", "") or "")[:80], ""])
    print(f"Wrote {out}. Fill the 'tier' column with 0..5 per the JD:")
    print("  5=perfect fit  4=strong  3=relevant  2=adjacent  1=weak  0=honeypot/wrong-role")
    print("  (You and Rutul label ~40-60 by hand; this is your blind-scoring ground truth.)")


def load_labels(path):
    labels = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = (row.get("tier") or "").strip()
            if t != "":
                labels[row["candidate_id"]] = int(float(t))
    return labels


# ──────────────────────────────────────────────────────────────────────────────
# 2. SCORE vs LABELS
# ──────────────────────────────────────────────────────────────────────────────

def score_against_labels(candidates, labels):
    rows, _ = R.rank_all(candidates, top_n=len(candidates))
    ranked = [labels.get(r["candidate_id"], 0) for r in rows]
    print(f"Labeled candidates: {len(labels)} / {len(candidates)}")
    print(f"  NDCG@10 : {ndcg_at_k(ranked, 10):.4f}   (50% of composite)")
    print(f"  NDCG@50 : {ndcg_at_k(ranked, 50):.4f}   (30%)")
    print(f"  MAP     : {average_precision(ranked):.4f}   (15%)")
    print(f"  P@10    : {precision_at_k(ranked, 10):.4f}   (5%)")
    composite = (0.50 * ndcg_at_k(ranked, 10) + 0.30 * ndcg_at_k(ranked, 50)
                 + 0.15 * average_precision(ranked) + 0.05 * precision_at_k(ranked, 10))
    print(f"  ► COMPOSITE (proxy): {composite:.4f}")
    return composite


# ──────────────────────────────────────────────────────────────────────────────
# 3. ABLATION — is the top-10 stable under weight perturbation?
# ──────────────────────────────────────────────────────────────────────────────

def ablate(candidates):
    """Cache each candidate's scoring components ONCE, then re-weight instantly
    for every perturbation. Turns 19 full scoring passes into a single pass."""
    now = date(2026, 6, 1)
    print("Scoring once and caching components …")
    cache = []
    for c in candidates:
        _, tr = R.score_candidate(c, now)
        cache.append((c.get("candidate_id", ""), tr))

    def rerank_top10(weights):
        out = []
        for cid, tr in cache:
            base = sum(weights[k] * v for k, v in tr["pillars"].items())
            final = base * tr["role_mult"] * (1 - tr["penalty"]) * tr["behavior_mult"]
            if tr["honeypot"]:
                final = final * 0.01 - 0.2
            out.append((cid, final))
        out.sort(key=lambda x: (-x[1], x[0]))
        return [cid for cid, _ in out[:10]]

    orig = copy.deepcopy(R.WEIGHTS)
    base_top = rerank_top10(orig)
    print("Baseline top-10:", base_top)
    print("\nPerturbing each weight ±40%, measuring top-10 overlap (higher=more stable):")
    worst = 1.0
    for k in orig:
        for delta in (0.6, 1.4):
            w = copy.deepcopy(orig)
            w[k] *= delta
            top = rerank_top10(w)
            overlap = len(set(top) & set(base_top)) / 10.0
            worst = min(worst, overlap)
            print(f"  {k:<22} ×{delta:<4} -> top-10 overlap {overlap:.0%}")
    print(f"\n  Worst-case overlap: {worst:.0%}  "
          f"({'STABLE' if worst >= 0.7 else 'FRAGILE — investigate'})")


# ──────────────────────────────────────────────────────────────────────────────
# 4. DUAL-SCORER AGREEMENT — independent strict rule-based scorer
# ──────────────────────────────────────────────────────────────────────────────

def strict_rule_score(c, now):
    """Independent scorer: hard gates only, minimal continuous blending.
    Disagreement with the weighted engine flags a candidate for manual review."""
    is_hp, _ = R.honeypot_flags(c)
    if is_hp:
        return -1.0
    rmult, rlabel = R.role_fit(c)
    if rlabel in ("non_technical", "non_target_tech"):
        return 0.0
    text = R._candidate_text(c).lower()
    dscore, _, cv_heavy = R.domain_evidence(c, text)
    yoe = float(c.get("profile", {}).get("years_of_experience", 0) or 0)
    sen = 1.0 if 5 <= yoe <= 9 else (0.5 if 4 <= yoe <= 11 else 0.2)
    bmult, _ = R.behavioral_multiplier(c, now)
    # strict: domain evidence dominates, hard role gate, behavioral as gate
    return rmult * (0.7 * dscore + 0.3 * sen) * bmult


def dual_agreement(candidates):
    now = date(2026, 6, 1)
    eng_rows, _ = R.rank_all(candidates, top_n=10)
    eng_top = [r["candidate_id"] for r in eng_rows]
    strict = sorted(candidates, key=lambda c: -strict_rule_score(c, now))
    strict_top = [c["candidate_id"] for c in strict[:10]]
    agree = set(eng_top) & set(strict_top)
    print("Weighted engine top-10:", eng_top)
    print("Strict rule top-10:    ", strict_top)
    print(f"\n  Agreement: {len(agree)}/10 high-confidence -> {sorted(agree)}")
    disagree = set(eng_top) ^ set(strict_top)
    if disagree:
        print(f"  Review (disagreements): {sorted(disagree)}")


# ──────────────────────────────────────────────────────────────────────────────
# 5. ADVERSARIAL SELF-TEST — inject traps, verify they're floored
# ──────────────────────────────────────────────────────────────────────────────

def selftest(candidates):
    now = date(2026, 6, 1)
    # synthetic keyword-stuffer: HR Manager loaded with AI skills
    stuffer = {
        "candidate_id": "TEST_STUFFER", "profile": {
            "anonymized_name": "Test Stuffer", "headline": "HR Manager | RAG, LLM, Embeddings, Retrieval",
            "summary": "RAG retrieval ranking embeddings vector search NDCG fine-tuning LLM transformers.",
            "current_title": "HR Manager", "years_of_experience": 7, "current_company": "X",
            "current_company_size": "201-500", "current_industry": "HR", "location": "Pune", "country": "India"},
        "career_history": [{"company": "X", "title": "HR Manager", "start_date": "2019-01-01",
            "end_date": None, "duration_months": 88, "is_current": True, "industry": "HR",
            "company_size": "201-500", "description": "RAG retrieval ranking embeddings vector search."}],
        "education": [], "skills": [{"name": n, "proficiency": "expert", "endorsements": 50,
            "duration_months": 40} for n in ["RAG", "LLM", "Embeddings", "Retrieval", "NLP", "Ranking"]],
        "redrob_signals": {"last_active_date": "2026-05-30", "recruiter_response_rate": 0.9,
            "interview_completion_rate": 0.9, "open_to_work_flag": True, "profile_completeness_score": 95,
            "github_activity_score": 80, "willing_to_relocate": True, "notice_period_days": 15,
            "expected_salary_range_inr_lpa": {"min": 30, "max": 50}, "skill_assessment_scores": {}}}
    # synthetic honeypot: expert in many skills with 0 months used
    honeypot = copy.deepcopy(stuffer)
    honeypot["candidate_id"] = "TEST_HONEYPOT"
    honeypot["profile"]["current_title"] = "ML Engineer"
    honeypot["profile"]["headline"] = "Senior ML Engineer | Retrieval, Ranking, RAG"
    honeypot["career_history"][0]["title"] = "ML Engineer"
    honeypot["skills"] = [{"name": n, "proficiency": "expert", "endorsements": 5,
        "duration_months": 0} for n in ["RAG", "Retrieval", "Ranking", "LLM", "NLP",
        "Embeddings", "PyTorch", "Search", "Vector DB", "MLOps"]]

    pool = candidates + [stuffer, honeypot]
    rows, _ = R.rank_all(pool, top_n=len(pool))
    pos = {r["candidate_id"]: (r["rank"], r["_tier"], r["_trace"]["honeypot"]) for r in rows}
    n = len(pool)
    print(f"Injected 2 traps into pool of {n}.")
    for tid in ("TEST_STUFFER", "TEST_HONEYPOT"):
        rk, ti, hp = pos[tid]
        bottom = rk > n * 0.5
        print(f"  {tid:<14} rank {rk}/{n}  tier {ti}  honeypot_flag={hp}  "
              f"-> {'FLOORED ✓' if (bottom or ti == 0) else 'NOT FLOORED ✗ — FIX'}")


# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("make-labels", "score", "ablate", "selftest", "dual"):
        s = sub.add_parser(name)
        s.add_argument("--candidates", required=True)
        if name == "make-labels":
            s.add_argument("--out", default="labels_template.csv")
        if name == "score":
            s.add_argument("--labels", required=True)
    args = ap.parse_args()

    cands = R.load_candidates(args.candidates)
    if args.cmd == "make-labels":
        make_labels(cands, args.out)
    elif args.cmd == "score":
        score_against_labels(cands, load_labels(args.labels))
    elif args.cmd == "ablate":
        ablate(cands)
    elif args.cmd == "dual":
        dual_agreement(cands)
    elif args.cmd == "selftest":
        selftest(cands)


if __name__ == "__main__":
    main()
