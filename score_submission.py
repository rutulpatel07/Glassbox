#!/usr/bin/env python3
"""
score_submission.py — Measure the Redrob competition metric on ANY submission CSV
=================================================================================
Computes:  0.50*NDCG@10 + 0.30*NDCG@50 + 0.15*MAP + 0.05*P@10
against your hand labels, so you can compare two submissions head-to-head.

Usage:
  python3 score_submission.py --submission submission_v3_calibrated.csv --labels labels_filled.csv
  python3 score_submission.py --submission submission_ltr.csv           --labels labels_filled.csv

Submission CSV must have columns: candidate_id, rank (or be in ranked order)
Labels CSV must have columns:      candidate_id, tier   (tier 0-5, 5 = best)

Requirements: pip install pandas numpy
"""

import argparse
import numpy as np
import pandas as pd

# Candidates with tier >= this count as "relevant" for MAP and P@10.
# Tier 4 = strong fit, Tier 5 = perfect fit. NDCG (80% of the score) does NOT
# depend on this threshold — only MAP and P@10 (20% combined) do.
REL_THRESHOLD = 4


def dcg(relevances):
    """Discounted Cumulative Gain with exponential gain (2^rel - 1)."""
    relevances = np.asarray(relevances, dtype=float)
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return np.sum((2 ** relevances - 1) / discounts)


def ndcg_at_k(ranked_rels, all_rels, k):
    """NDCG@k: ranked_rels in submission order, all_rels = every label (for ideal)."""
    actual = dcg(ranked_rels[:k])
    ideal_rels = sorted(all_rels, reverse=True)[:k]
    ideal = dcg(ideal_rels)
    return actual / ideal if ideal > 0 else 0.0


def average_precision(ranked_rels, total_relevant):
    """AP: precision averaged at each relevant hit, divided by total relevant."""
    if total_relevant == 0:
        return 0.0
    hits = 0
    score = 0.0
    for i, rel in enumerate(ranked_rels, start=1):
        if rel >= REL_THRESHOLD:
            hits += 1
            score += hits / i
    return score / total_relevant


def precision_at_k(ranked_rels, k):
    """Fraction of top-k that are relevant."""
    topk = ranked_rels[:k]
    if len(topk) == 0:
        return 0.0
    return sum(1 for r in topk if r >= REL_THRESHOLD) / k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--labels", required=True)
    args = parser.parse_args()

    sub = pd.read_csv(args.submission)
    lab = pd.read_csv(args.labels)

    # Sort submission by rank if the column exists; else assume file order
    if "rank" in sub.columns:
        sub = sub.sort_values("rank").reset_index(drop=True)

    tier_map = dict(zip(lab["candidate_id"], lab["tier"]))

    # Relevance of each ranked candidate (unjudged candidates count as 0)
    ranked_rels = [tier_map.get(cid, 0) for cid in sub["candidate_id"]]

    all_rels = list(lab["tier"].values)              # universe for ideal DCG
    total_relevant = int((lab["tier"] >= REL_THRESHOLD).sum())

    ndcg10 = ndcg_at_k(ranked_rels, all_rels, 10)
    ndcg50 = ndcg_at_k(ranked_rels, all_rels, 50)
    ap = average_precision(ranked_rels, total_relevant)
    p10 = precision_at_k(ranked_rels, 10)

    composite = 0.50 * ndcg10 + 0.30 * ndcg50 + 0.15 * ap + 0.05 * p10

    judged_in_top100 = sum(1 for cid in sub["candidate_id"] if cid in tier_map)

    print("=" * 52)
    print(f"  Scoring: {args.submission}")
    print("=" * 52)
    print(f"  NDCG@10 : {ndcg10:.4f}   (weight 0.50)")
    print(f"  NDCG@50 : {ndcg50:.4f}   (weight 0.30)")
    print(f"  MAP     : {ap:.4f}   (weight 0.15, relevant = tier>={REL_THRESHOLD})")
    print(f"  P@10    : {p10:.4f}   (weight 0.05)")
    print("  " + "-" * 48)
    print(f"  COMPOSITE SCORE: {composite:.4f}")
    print("=" * 52)
    print(f"  Labeled candidates present in this top-100: {judged_in_top100}/100")
    if judged_in_top100 < 100:
        print(f"  ⚠  {100 - judged_in_top100} ranked candidates are unjudged (counted as tier 0).")
    print()


if __name__ == "__main__":
    main()
