#!/usr/bin/env python3
"""
tune.py — auto-fit the tier-prediction thresholds to YOUR hand labels.
=======================================================================
The engine maps a candidate's final score -> tier 0..5 using cutoffs. Those
cutoffs were set by judgment. Once you've hand-labeled real candidates
(validation_harness.py make-labels), this script finds the cutoffs that best
reproduce your labels, so the tiers reflect reality instead of a guess.

It does NOT change how candidates are scored or ranked — only the score->tier
mapping used for reasoning tone and the NDCG self-check. The ranking order is
unchanged (it's by raw score). Safe to run; prints suggested cutoffs to paste
into rank.py predict_tier().

Usage
-----
    python tune.py --candidates sample_candidates.json --labels labels_filled.csv
"""

import argparse
from datetime import date
from itertools import product

import numpy as np

import rank as R
from validation_harness import load_labels, ndcg_at_k, average_precision, precision_at_k


def tune(candidates, labels):
    now = date(2026, 6, 1)
    scored = []
    for c in candidates:
        cid = c.get("candidate_id", "")
        if cid in labels:
            final, _ = R.score_candidate(c, now)
            scored.append((cid, final, labels[cid]))
    if len(scored) < 10:
        print(f"Only {len(scored)} labeled candidates found — label more (~40-60) for a reliable fit.")
        return

    finals = np.array([s[1] for s in scored])
    lo, hi = float(finals.min()), float(finals.max())
    grid = np.linspace(lo, hi, 12)

    def assign(cutoffs, f):
        c1, c2, c3, c4, c5 = cutoffs
        if f >= c5: return 5
        if f >= c4: return 4
        if f >= c3: return 3
        if f >= c2: return 2
        if f >= c1: return 1
        return 0

    # search monotone cutoff sets, maximize agreement (weighted by tier distance)
    best, best_acc = None, -1
    cand_cuts = sorted(set(np.round(grid, 4)))
    for combo in product(cand_cuts, repeat=5):
        if not (combo[0] < combo[1] < combo[2] < combo[3] < combo[4]):
            continue
        err = sum(abs(assign(combo, f) - t) for _, f, t in scored)
        acc = 1 - err / (5 * len(scored))
        if acc > best_acc:
            best_acc, best = acc, combo
    print(f"Labeled set: {len(scored)} candidates")
    print(f"Best tier-distance accuracy: {best_acc:.3f}")
    print("Suggested predict_tier() cutoffs (paste into rank.py):")
    print(f"  tier5 if f >= {best[4]:.3f}")
    print(f"  tier4 if f >= {best[3]:.3f}")
    print(f"  tier3 if f >= {best[2]:.3f}")
    print(f"  tier2 if f >= {best[1]:.3f}")
    print(f"  tier1 if f >= {best[0]:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--labels", required=True)
    args = ap.parse_args()
    tune(R.load_candidates(args.candidates), load_labels(args.labels))


if __name__ == "__main__":
    main()
