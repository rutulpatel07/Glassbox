#!/usr/bin/env python3
"""
experiments.py — Variant A-W cascade experiment suite for rank.py
=================================================================
Imported only when rank.py is called with --experiments.  Contains all
two-stage cascade builder functions and the full experiment driver.

Usage (via rank.py):
    python rank.py --candidates candidates.jsonl --out submission.csv --experiments
"""

import csv as _csv
import math

import numpy as np

from rank import (
    _cascade_sort_key,
    _make_cascade_rows,
    predict_tier,
    tier5_signal,
    write_csv,
)


# ──────────────────────────────────────────────────────────────────────────────
# CASCADE BUILDER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def build_cascade_variants(scored, K=10, top_n=100):
    """
    Variant A (protected K=10): top-K by final_order locked in; fill to top_n by
    final_select; rank pool by final_order.
    Variant B (pure K=0): top-top_n by final_select; rank by final_order.
    scored items are 5-tuples: (candidate, final_order, trace, confidence, final_select).
    """
    order_idx  = sorted(range(len(scored)), key=lambda i: _cascade_sort_key(scored[i]))
    select_idx = sorted(range(len(scored)), key=lambda i: -scored[i][4])

    # Variant A
    protected_ids = set(scored[i][0].get("candidate_id", "") for i in order_idx[:K])
    pool_a = list(order_idx[:K])
    seen   = set(protected_ids)
    for i in select_idx:
        if len(pool_a) == top_n:
            break
        cid = scored[i][0].get("candidate_id", "")
        if cid not in seen:
            pool_a.append(i)
            seen.add(cid)
    rows_a = _make_cascade_rows(
        sorted([scored[i] for i in pool_a], key=_cascade_sort_key), top_n)

    # Variant B
    pool_b = select_idx[:top_n]
    rows_b = _make_cascade_rows(
        sorted([scored[i] for i in pool_b], key=_cascade_sort_key), top_n)

    return rows_a, rows_b


def build_pool_expand_variant(scored, stage1_n, final_n=100):
    """
    Stage 1: take top-stage1_n by final_select.
    Stage 2: sort those by final_order, return top-final_n rows + the stage-1 pool.
    """
    select_idx = sorted(range(len(scored)), key=lambda i: -scored[i][4])
    stage1 = [scored[i] for i in select_idx[:stage1_n]]
    rows = _make_cascade_rows(sorted(stage1, key=_cascade_sort_key), final_n)
    return rows, stage1


def build_twotier_variant(scored, head_k, tail_key_fn=None, pool_n=100, final_n=100):
    """
    Pool: top-pool_n by final_select.
    Positions 1..head_k  : ordered by final_order (behavior-weighted).
    Positions head_k+1.. : ordered by tail_key_fn (default: final_select desc).
    Score col: final_order, capped monotone non-increasing.
    """
    select_idx = sorted(range(len(scored)), key=lambda i: -scored[i][4])
    pool = [scored[i] for i in select_idx[:pool_n]]
    order_sorted = sorted(pool, key=_cascade_sort_key)
    head = order_sorted[:head_k]
    if tail_key_fn is None:
        tail_key_fn = lambda item: -item[4]
    tail = sorted(order_sorted[head_k:], key=tail_key_fn)
    return _make_cascade_rows(head + tail, final_n)


def build_ordering_variant(scored, order_key_fn, pool_n=100, final_n=100):
    """
    Pool: top-pool_n by final_select (Variant B's pool).
    Order: by order_key_fn(scored_item) ascending (negate inside fn for desc).
    Score column in CSV: final_order (item[1]), capped monotone non-increasing.
    """
    select_idx = sorted(range(len(scored)), key=lambda i: -scored[i][4])
    pool = [scored[i] for i in select_idx[:pool_n]]
    return _make_cascade_rows(sorted(pool, key=order_key_fn), final_n)


# ──────────────────────────────────────────────────────────────────────────────
# INLINE SCORER (mirrors score_submission.py with REL_THRESHOLD=4)
# ──────────────────────────────────────────────────────────────────────────────

def _score_against_labels(rows, labels_path):
    tier_map, all_tiers = {}, []
    with open(labels_path, newline="", encoding="utf-8") as f:
        for rec in _csv.DictReader(f):
            t = int(rec["tier"])
            tier_map[rec["candidate_id"]] = t
            all_tiers.append(t)

    ranked_rels = [tier_map.get(r["candidate_id"], 0) for r in rows]
    total_rel   = sum(1 for t in all_tiers if t >= 4)

    def _dcg(rels):
        r = np.asarray(rels, dtype=float)
        return float(np.sum((2**r - 1) / np.log2(np.arange(2, len(r)+2))))

    def _ndcg(ranked, k):
        ideal = _dcg(sorted(all_tiers, reverse=True)[:k])
        return _dcg(ranked[:k]) / ideal if ideal else 0.0

    def _ap(ranked):
        if not total_rel:
            return 0.0
        hits = s = 0
        for i, r in enumerate(ranked, 1):
            if r >= 4:
                hits += 1
                s += hits / i
        return s / total_rel

    n10 = _ndcg(ranked_rels, 10)
    n50 = _ndcg(ranked_rels, 50)
    ap  = _ap(ranked_rels)
    p10 = sum(1 for r in ranked_rels[:10] if r >= 4) / 10.0
    return {"ndcg10": n10, "ndcg50": n50, "map": ap, "p10": p10,
            "composite": 0.50*n10 + 0.30*n50 + 0.15*ap + 0.05*p10}


# ──────────────────────────────────────────────────────────────────────────────
# MAIN EXPERIMENT DRIVER
# ──────────────────────────────────────────────────────────────────────────────

def run_all_experiments(scored, labels_path="labels_filled 500.csv",
                        cascade_K=10, top_n=100):
    """Run all A-W cascade experiments against labels_path."""

    BASELINE    = 0.6532
    BASELINE_V2 = 0.6933
    NDCG10_MIN  = 0.7699

    def _t45(r):
        return sum(1 for x in r if x["_tier"] >= 4)

    def _t45_pool(pl):
        return sum(1 for item in pl if predict_tier(item[2]) >= 4)

    # ── Variants A and B ───────────────────────────────────────────────────────
    print("\nBuilding cascade variants A/B …")
    rows_a, rows_b = build_cascade_variants(scored, K=cascade_K, top_n=top_n)
    sc_a = _score_against_labels(rows_a, labels_path)
    sc_b = _score_against_labels(rows_b, labels_path)

    print("\n" + "=" * 60)
    print("  CASCADE — VARIANTS A / B  (baseline 0.6532)")
    print("=" * 60)
    for lbl, sc, rws in [("A  protected K=10", sc_a, rows_a),
                          ("B  pure K=0",       sc_b, rows_b)]:
        print(f"  Variant {lbl}")
        print(f"    NDCG@10  : {sc['ndcg10']:.4f}  (weight 0.50)")
        print(f"    NDCG@50  : {sc['ndcg50']:.4f}  (weight 0.30)")
        print(f"    MAP      : {sc['map']:.4f}  (weight 0.15)")
        print(f"    P@10     : {sc['p10']:.4f}  (weight 0.05)")
        print(f"    COMPOSITE: {sc['composite']:.4f}")
        print(f"    Tier≥4 in top-100: {_t45(rws)}")
        print()

    best_ab = max([(sc_a, rows_a, "A"), (sc_b, rows_b, "B")],
                  key=lambda x: x[0]["composite"])
    if best_ab[0]["composite"] > BASELINE:
        write_csv(best_ab[1], "submission_cascade.csv")
        print(f"  A/B winner: Variant {best_ab[2]}  "
              f"COMPOSITE {best_ab[0]['composite']:.4f} — wrote submission_cascade.csv")
    print("=" * 60)

    # ── Variants C and D (pool expansion) ─────────────────────────────────────
    print("\nBuilding pool-expansion variants C/D …")
    rows_c, pool_c = build_pool_expand_variant(scored, stage1_n=200, final_n=top_n)
    rows_d, pool_d = build_pool_expand_variant(scored, stage1_n=500, final_n=top_n)
    sc_c = _score_against_labels(rows_c, labels_path)
    sc_d = _score_against_labels(rows_d, labels_path)

    print("\n" + "=" * 60)
    print(f"  CASCADE — VARIANTS C / D  (baseline {BASELINE_V2})")
    print("=" * 60)
    for lbl, sc, rws, pool, s1n in [
            ("C  stage1=200", sc_c, rows_c, pool_c, 200),
            ("D  stage1=500", sc_d, rows_d, pool_d, 500)]:
        ndcg_flag = "✓" if sc["ndcg10"] >= NDCG10_MIN else "✗"
        print(f"  Variant {lbl}")
        print(f"    NDCG@10  : {sc['ndcg10']:.4f}  (weight 0.50)  [{ndcg_flag} ≥{NDCG10_MIN}]")
        print(f"    NDCG@50  : {sc['ndcg50']:.4f}  (weight 0.30)")
        print(f"    MAP      : {sc['map']:.4f}  (weight 0.15)")
        print(f"    P@10     : {sc['p10']:.4f}  (weight 0.05)")
        print(f"    COMPOSITE: {sc['composite']:.4f}  baseline {BASELINE_V2}")
        print(f"    Tier≥4 in stage-1 pool ({s1n:>3}): {_t45_pool(pool)}")
        print(f"    Tier≥4 in final top-100:       {_t45(rws)}")
        print()

    best_cd = max([(sc_c, rows_c, "C"), (sc_d, rows_d, "D")],
                  key=lambda x: x[0]["composite"])
    best_cd_sc, best_cd_rows, best_cd_lbl = best_cd

    print("=" * 60)
    if best_cd_sc["composite"] > BASELINE_V2 and best_cd_sc["ndcg10"] >= NDCG10_MIN:
        write_csv(best_cd_rows, "submission_cascade_v2.csv")
        print(f"  C/D winner: Variant {best_cd_lbl}  "
              f"COMPOSITE {best_cd_sc['composite']:.4f} > {BASELINE_V2}  "
              f"NDCG@10 {best_cd_sc['ndcg10']:.4f} ≥ {NDCG10_MIN}")
        print(f"  Wrote → submission_cascade_v2.csv")
    else:
        reasons = []
        if best_cd_sc["composite"] <= BASELINE_V2:
            reasons.append(f"COMPOSITE {best_cd_sc['composite']:.4f} ≤ {BASELINE_V2}")
        if best_cd_sc["ndcg10"] < NDCG10_MIN:
            reasons.append(f"NDCG@10 {best_cd_sc['ndcg10']:.4f} < {NDCG10_MIN}")
        print(f"  Best C/D: Variant {best_cd_lbl} — gate not cleared ({'; '.join(reasons)})")
        print("  submission_cascade_v2.csv NOT written.")
    print("=" * 60)

    # ── Variants E, F, G (Stage-2 ordering on B's pool) ───────────────────────
    print("\nBuilding Stage-2 ordering variants E/F/G …")
    rows_e = build_ordering_variant(scored, lambda item: -item[4])
    rows_f = build_ordering_variant(scored,
        lambda item: -(item[4] * (0.95 + 0.05 * item[2]["behavior"].get("behavior_core", 0))))
    rows_g = build_ordering_variant(scored, lambda item: -item[2]["base_fit"])

    sc_e = _score_against_labels(rows_e, labels_path)
    sc_f = _score_against_labels(rows_f, labels_path)
    sc_g = _score_against_labels(rows_g, labels_path)

    print("\n" + "=" * 60)
    print(f"  CASCADE — VARIANTS E / F / G  (baseline {BASELINE_V2})")
    print("=" * 60)
    for lbl, sc, rws in [
            ("E  order=final_select",              sc_e, rows_e),
            ("F  order=final_select*(0.95+0.05b)", sc_f, rows_f),
            ("G  order=base_fit",                  sc_g, rows_g)]:
        ndcg_flag = "✓" if sc["ndcg10"] >= NDCG10_MIN else "✗"
        print(f"  Variant {lbl}")
        print(f"    NDCG@10  : {sc['ndcg10']:.4f}  (weight 0.50)  [{ndcg_flag} ≥{NDCG10_MIN}]")
        print(f"    NDCG@50  : {sc['ndcg50']:.4f}  (weight 0.30)")
        print(f"    MAP      : {sc['map']:.4f}  (weight 0.15)")
        print(f"    P@10     : {sc['p10']:.4f}  (weight 0.05)")
        print(f"    COMPOSITE: {sc['composite']:.4f}  baseline {BASELINE_V2}")
        print()

    best_efg_sc, best_efg_rows, best_efg_lbl = max(
        [(sc_e, rows_e, "E"), (sc_f, rows_f, "F"), (sc_g, rows_g, "G")],
        key=lambda x: x[0]["composite"])

    print("=" * 60)
    if best_efg_sc["composite"] > BASELINE_V2 and best_efg_sc["ndcg10"] >= NDCG10_MIN:
        write_csv(best_efg_rows, "submission_cascade_v2.csv")
        print(f"  E/F/G winner: Variant {best_efg_lbl}  "
              f"COMPOSITE {best_efg_sc['composite']:.4f} > {BASELINE_V2}  "
              f"NDCG@10 {best_efg_sc['ndcg10']:.4f} ≥ {NDCG10_MIN}")
        print(f"  Wrote → submission_cascade_v2.csv")
    else:
        reasons = []
        if best_efg_sc["composite"] <= BASELINE_V2:
            reasons.append(f"COMPOSITE {best_efg_sc['composite']:.4f} ≤ {BASELINE_V2}")
        if best_efg_sc["ndcg10"] < NDCG10_MIN:
            reasons.append(f"NDCG@10 {best_efg_sc['ndcg10']:.4f} < {NDCG10_MIN}")
        print(f"  Best E/F/G: Variant {best_efg_lbl} — "
              f"gate not cleared ({'; '.join(reasons)})")
        print("  submission_cascade_v2.csv NOT written.")
    print("=" * 60)

    # ── Variants H, I (two-tier rank-dependent ordering) ──────────────────────
    print("\nBuilding two-tier ordering variants H/I …")
    rows_h = build_twotier_variant(scored, head_k=15)
    rows_i = build_twotier_variant(scored, head_k=10)
    sc_h = _score_against_labels(rows_h, labels_path)
    sc_i = _score_against_labels(rows_i, labels_path)

    print("\n" + "=" * 60)
    print(f"  CASCADE — VARIANTS H / I  (baseline {BASELINE_V2})")
    print("=" * 60)
    for lbl, sc, rws in [
            ("H  head=15 final_order / tail=85 final_select", sc_h, rows_h),
            ("I  head=10 final_order / tail=90 final_select", sc_i, rows_i)]:
        ndcg_flag = "✓" if sc["ndcg10"] >= NDCG10_MIN else "✗"
        print(f"  Variant {lbl}")
        print(f"    NDCG@10  : {sc['ndcg10']:.4f}  (weight 0.50)  [{ndcg_flag} ≥{NDCG10_MIN}]")
        print(f"    NDCG@50  : {sc['ndcg50']:.4f}  (weight 0.30)")
        print(f"    MAP      : {sc['map']:.4f}  (weight 0.15)")
        print(f"    P@10     : {sc['p10']:.4f}  (weight 0.05)")
        print(f"    COMPOSITE: {sc['composite']:.4f}  baseline {BASELINE_V2}")
        print()

    best_hi_sc, best_hi_rows, best_hi_lbl = max(
        [(sc_h, rows_h, "H"), (sc_i, rows_i, "I")],
        key=lambda x: x[0]["composite"])

    print("=" * 60)
    if best_hi_sc["composite"] > BASELINE_V2 and round(best_hi_sc["ndcg10"], 4) >= NDCG10_MIN:
        write_csv(best_hi_rows, "submission_cascade_v2.csv")
        print(f"  H/I winner: Variant {best_hi_lbl}  "
              f"COMPOSITE {best_hi_sc['composite']:.4f} > {BASELINE_V2}  "
              f"NDCG@10 {best_hi_sc['ndcg10']:.4f} ≥ {NDCG10_MIN}")
        print(f"  Wrote → submission_cascade_v2.csv")
    else:
        reasons = []
        if best_hi_sc["composite"] <= BASELINE_V2:
            reasons.append(f"COMPOSITE {best_hi_sc['composite']:.4f} ≤ {BASELINE_V2}")
        if best_hi_sc["ndcg10"] < NDCG10_MIN:
            reasons.append(f"NDCG@10 {best_hi_sc['ndcg10']:.4f} < {NDCG10_MIN}")
        print(f"  Best H/I: Variant {best_hi_lbl} — "
              f"gate not cleared ({'; '.join(reasons)})")
        print("  submission_cascade_v2.csv NOT written.")
    print("=" * 60)

    # ── J/K/L/M tail-key experiment + head boundary sweep ─────────────────────
    BASELINE_V3 = 0.7081

    TAIL_SPECS = [
        ("J", "tail=final_select        ← H",
         lambda item: -item[4]),
        ("K", "tail=base_fit",
         lambda item: -item[2]["base_fit"]),
        ("L", "tail=domain_evidence",
         lambda item: -item[2]["pillars"]["domain_evidence"]),
        ("M", "tail=0.6·fs + 0.4·bf",
         lambda item: -(0.6 * item[4] + 0.4 * item[2]["base_fit"])),
    ]

    print("\nBuilding tail-key variants J/K/L/M (head=15 fixed) …")
    tail_res = {}
    for vname, vdesc, vfn in TAIL_SPECS:
        rws = build_twotier_variant(scored, head_k=15, tail_key_fn=vfn)
        sc  = _score_against_labels(rws, labels_path)
        tail_res[vname] = (sc, rws, vfn, vdesc)

    best_tail_name = max(tail_res, key=lambda k: tail_res[k][0]["composite"])
    best_tail_fn   = tail_res[best_tail_name][2]
    print(f"  Best tail key: {best_tail_name}")

    print("Building head boundary sweep {12, 15, 18} with best tail key …")
    head_res = {}
    for hk in [12, 15, 18]:
        lbl = f"{best_tail_name}+h{hk}"
        rws = build_twotier_variant(scored, head_k=hk, tail_key_fn=best_tail_fn)
        sc  = _score_against_labels(rws, labels_path)
        head_res[lbl] = (sc, rws, hk)

    W = 80
    print("\n" + "=" * W)
    print(f"  {'Variant':<43} NDCG@10  NDCG@50     MAP    P@10    COMP")
    print("  " + "-" * (W - 2))

    all_results = []

    print("  tail-key sweep  (head=15 fixed)")
    for vname, vdesc, _ in TAIL_SPECS:
        sc, rws, *_ = tail_res[vname]
        flag = "✓" if round(sc["ndcg10"], 4) >= NDCG10_MIN else "✗"
        rlbl = f"{vname}  {vdesc}"
        print(f"  {rlbl:<43} {sc['ndcg10']:.4f}{flag} {sc['ndcg50']:.4f}  "
              f"{sc['map']:.4f}  {sc['p10']:.4f}  {sc['composite']:.4f}")
        all_results.append((sc, rws, vname))

    print(f"\n  head boundary sweep  (tail={best_tail_name})")
    for lbl in sorted(head_res, key=lambda k: head_res[k][2]):
        sc, rws, hk = head_res[lbl]
        flag = "✓" if round(sc["ndcg10"], 4) >= NDCG10_MIN else "✗"
        note = "  ← H" if hk == 15 and best_tail_name == "J" else ""
        print(f"  {lbl + note:<43} {sc['ndcg10']:.4f}{flag} {sc['ndcg50']:.4f}  "
              f"{sc['map']:.4f}  {sc['p10']:.4f}  {sc['composite']:.4f}")
        all_results.append((sc, rws, lbl))

    print("=" * W)

    best_v_sc, best_v_rows, best_v_lbl = max(all_results, key=lambda x: x[0]["composite"])
    print(f"\n  Best candidate: {best_v_lbl}  COMPOSITE {best_v_sc['composite']:.4f}")
    if best_v_sc["composite"] > BASELINE_V3 and round(best_v_sc["ndcg10"], 4) >= NDCG10_MIN:
        write_csv(best_v_rows, "submission_final.csv")
        print(f"  Gate cleared: COMPOSITE {best_v_sc['composite']:.4f} > {BASELINE_V3}  "
              f"NDCG@10 {best_v_sc['ndcg10']:.4f} ≥ {NDCG10_MIN}")
        print(f"  Wrote → submission_final.csv")
    else:
        reasons = []
        if best_v_sc["composite"] <= BASELINE_V3:
            reasons.append(f"COMPOSITE {best_v_sc['composite']:.4f} ≤ {BASELINE_V3}")
        if round(best_v_sc["ndcg10"], 4) < NDCG10_MIN:
            reasons.append(f"NDCG@10 {best_v_sc['ndcg10']:.4f} < {NDCG10_MIN}")
        print(f"  Gate not cleared ({'; '.join(reasons)}) — submission_final.csv NOT written.")

    # ── Stage-1 pool-expansion experiment (Variants N / O / P) ────────────────
    BASELINE_L = 0.7101

    tail_l = lambda item: -item[2]["pillars"]["domain_evidence"]

    select_idx_all = sorted(range(len(scored)), key=lambda i: -scored[i][4])

    POOL_SPECS = [("N", 150), ("O", 200), ("P", 300)]
    nop_results = []
    for vname, pn in POOL_SPECS:
        pool = [scored[i] for i in select_idx_all[:pn]]
        t45_stage1 = _t45_pool(pool)
        rows_v = build_twotier_variant(scored, head_k=15, tail_key_fn=tail_l, pool_n=pn)
        sc_v   = _score_against_labels(rows_v, labels_path)
        t45_top100 = _t45(rows_v)
        nop_results.append((vname, pn, sc_v, rows_v, t45_stage1, t45_top100))

    print("\n" + "=" * W)
    print(f"  POOL EXPANSION — VARIANTS N / O / P  (baseline {BASELINE_L})")
    print(f"  Stage-2: head=15 final_order / tail=domain_evidence (Variant L recipe)")
    print("  " + "-" * (W - 2))
    print(f"  {'Variant':<18} NDCG@10  NDCG@50     MAP    P@10    COMP"
          f"   t≥4(s1) t≥4(top100)")
    print("  " + "-" * (W - 2))

    for vname, pn, sc, rws, t45s1, t45top in nop_results:
        flag = "✓" if round(sc["ndcg10"], 4) >= NDCG10_MIN else "✗"
        gate = ">" if sc["composite"] > BASELINE_L else "≤"
        lbl  = f"{vname}  pool={pn}"
        print(f"  {lbl:<18} {sc['ndcg10']:.4f}{flag} {sc['ndcg50']:.4f}  "
              f"{sc['map']:.4f}  {sc['p10']:.4f}  {sc['composite']:.4f}{gate}  "
              f"{t45s1:>7}  {t45top:>10}")

    print("=" * W)

    best_nop = max(nop_results, key=lambda x: x[2]["composite"])
    vname_b, pn_b, sc_b, rows_b, t45s1_b, t45top_b = best_nop
    print(f"\n  Best: Variant {vname_b} (pool={pn_b})  "
          f"COMPOSITE {sc_b['composite']:.4f}  NDCG@10 {sc_b['ndcg10']:.4f}")

    if sc_b["composite"] > BASELINE_L and round(sc_b["ndcg10"], 4) >= NDCG10_MIN:
        write_csv(rows_b, "submission_final.csv")
        print(f"  Gate cleared: wrote → submission_final.csv")
    else:
        reasons = []
        if sc_b["composite"] <= BASELINE_L:
            reasons.append(f"COMPOSITE {sc_b['composite']:.4f} ≤ {BASELINE_L}")
        if round(sc_b["ndcg10"], 4) < NDCG10_MIN:
            reasons.append(f"NDCG@10 {sc_b['ndcg10']:.4f} < {NDCG10_MIN}")
        print(f"  Gate NOT cleared ({'; '.join(reasons)}) — submission_final.csv unchanged.")
    print("=" * W)

    # ── Head-ordering experiment (Variants Q / R / S) ─────────────────────────
    t5_map = {}
    for c_item, _fo, _tr, _cf, _fs in scored:
        cid = c_item.get("candidate_id", "")
        t5_map[cid] = tier5_signal(c_item)

    sel_idx = sorted(range(len(scored)), key=lambda i: -scored[i][4])
    pool    = [scored[i] for i in sel_idx[:100]]

    pool_by_order = sorted(pool, key=_cascade_sort_key)
    L_head = pool_by_order[:15]
    L_head_ids = {it[0].get("candidate_id", "") for it in L_head}
    L_tail = sorted(pool_by_order[15:],
                    key=lambda it: -it[2]["pillars"]["domain_evidence"])
    rows_l = _make_cascade_rows(L_head + L_tail, 100)
    sc_l   = _score_against_labels(rows_l, labels_path)
    L_top10_ids = [r["candidate_id"] for r in rows_l[:10]]

    fo_order_idx = sorted(range(100), key=lambda i: pool[i][1])
    fo_pct = [0.0] * 100
    for rank_i, idx in enumerate(fo_order_idx):
        fo_pct[idx] = rank_i / 99.0
    cid_to_fo_pct = {pool[i][0].get("candidate_id", ""): fo_pct[i] for i in range(100)}

    def _build_head_variant(pool, head_key_fn):
        ordered = sorted(pool, key=head_key_fn)
        head = ordered[:15]
        tail = sorted(ordered[15:], key=lambda it: -it[2]["pillars"]["domain_evidence"])
        return _make_cascade_rows(head + tail, 100)

    def q_key(it):
        return -(it[1] * (0.85 + 0.15 * t5_map.get(it[0].get("candidate_id", ""), 0.0)))

    def r_key(it):
        cid = it[0].get("candidate_id", "")
        return -(0.70 * cid_to_fo_pct.get(cid, 0.0) + 0.30 * t5_map.get(cid, 0.0))

    rows_q = _build_head_variant(pool, q_key)
    sc_q   = _score_against_labels(rows_q, labels_path)

    rows_r = _build_head_variant(pool, r_key)
    sc_r   = _score_against_labels(rows_r, labels_path)

    s_head = sorted(L_head, key=q_key)
    rows_s = _make_cascade_rows(s_head + L_tail, 100)
    sc_s   = _score_against_labels(rows_s, labels_path)

    tier_map = {}
    try:
        with open(labels_path, newline="", encoding="utf-8") as _f:
            for _rec in _csv.DictReader(_f):
                tier_map[_rec["candidate_id"]] = int(_rec["tier"])
    except Exception:
        pass

    W2 = 74
    print("\n" + "=" * W2)
    print("  HEAD-ORDERING EXPERIMENT — VARIANTS Q / R / S")
    print(f"  Fixed: pool=100 by final_select, tail by domain_evidence, head_k=15")
    print("  " + "-" * (W2 - 2))
    print(f"  {'Variant':<26} NDCG@10  NDCG@50     MAP   P@10  COMPOSITE")
    print("  " + "-" * (W2 - 2))

    for lbl, sc in [("L  (base, head=fo)",      sc_l),
                     ("Q  head=fo*(0.85+0.15t5)", sc_q),
                     ("R  head=0.70pct+0.30t5",  sc_r),
                     ("S  same15 reorder by Q",  sc_s)]:
        g   = ">" if sc["composite"] > BASELINE_L else "≤"
        p10 = "✓" if sc["p10"] >= 1.0 else "✗"
        print(f"  {lbl:<26} {sc['ndcg10']:.4f}  {sc['ndcg50']:.4f}  "
              f"{sc['map']:.4f}  {sc['p10']:.2f}{p10}  {sc['composite']:.4f}{g}")

    print("=" * W2)

    for vname, rows_v in [("Q", rows_q), ("R", rows_r), ("S", rows_s)]:
        top10_v = [r["candidate_id"] for r in rows_v[:10]]
        entered = [c for c in top10_v if c not in L_top10_ids]
        left    = [c for c in L_top10_ids if c not in top10_v]
        print(f"\n  Variant {vname} — top-10 vs L:")
        if not entered and not left:
            print("    composition unchanged (reordering within same 10)")
        if entered:
            print("    ENTERED: " + ", ".join(
                f"{c}(t{tier_map.get(c, '?')})" for c in entered))
        if left:
            print("    LEFT   : " + ", ".join(
                f"{c}(t{tier_map.get(c, '?')})" for c in left))
        print(f"    Top-10 order: " +
              " ".join(f"[{i+1}]{r['candidate_id']}(t{tier_map.get(r['candidate_id'],'?')})"
                       for i, r in enumerate(rows_v[:10])))
    print(f"\n  Variant L top-10: " +
          " ".join(f"[{i+1}]{cid}(t{tier_map.get(cid,'?')})"
                   for i, cid in enumerate(L_top10_ids)))

    valid = [(sc, rws, nm) for sc, rws, nm in
             [(sc_q, rows_q, "Q"), (sc_r, rows_r, "R"), (sc_s, rows_s, "S")]
             if sc["composite"] > BASELINE_L and sc["p10"] >= 1.0]

    print()
    if valid:
        best_sc, best_rows, best_lbl = max(valid, key=lambda x: x[0]["composite"])
        write_csv(best_rows, "submission_final.csv")
        print(f"  Winner: Variant {best_lbl}  COMPOSITE {best_sc['composite']:.4f} > {BASELINE_L}"
              f"  P@10 {best_sc['p10']:.2f}  → wrote submission_final.csv")
    else:
        print("  Gate NOT cleared — submission_final.csv unchanged.")
        for sc, _, nm in [(sc_q, rows_q, "Q"), (sc_r, rows_r, "R"), (sc_s, rows_s, "S")]:
            rs = []
            if sc["composite"] <= BASELINE_L:
                rs.append(f"COMP {sc['composite']:.4f}≤{BASELINE_L}")
            if sc["p10"] < 1.0:
                rs.append(f"P@10 {sc['p10']:.2f}<1.0")
            print(f"    {nm}: {', '.join(rs)}")
    print("=" * W2)

    # ── Head-zone expansion (Variants T / U) ──────────────────────────────────
    BASELINE_S = 0.7229

    def _t5_order_key(it):
        return -(it[1] * (0.85 + 0.15 * t5_map.get(it[0].get("candidate_id", ""), 0.0)))

    pool2 = [scored[i] for i in sorted(range(len(scored)), key=lambda i: -scored[i][4])[:100]]
    pool2_by_order = sorted(pool2, key=_cascade_sort_key)

    def _build_tu(head_k):
        head = sorted(pool2_by_order[:head_k], key=_t5_order_key)
        tail = sorted(pool2_by_order[head_k:],
                      key=lambda it: -it[2]["pillars"]["domain_evidence"])
        return _make_cascade_rows(head + tail, 100)

    rows_t = _build_tu(18)
    rows_u = _build_tu(20)
    sc_t   = _score_against_labels(rows_t, labels_path)
    sc_u   = _score_against_labels(rows_u, labels_path)

    rows_s_ref = _make_cascade_rows(
        sorted(pool2_by_order[:15], key=_t5_order_key) +
        sorted(pool2_by_order[15:], key=lambda it: -it[2]["pillars"]["domain_evidence"]),
        100)
    sc_s_ref = _score_against_labels(rows_s_ref, labels_path)

    W3 = 72
    print("\n" + "=" * W3)
    print("  HEAD-ZONE EXPANSION — VARIANTS T / U  (base: Variant S 0.7229)")
    print("  Fixed: pool=100 by final_select, tail by domain_evidence")
    print("  Head reorder key: final_order * (0.85 + 0.15*t5)  [same as S]")
    print("  " + "-" * (W3 - 2))
    print(f"  {'Variant':<24} NDCG@10  NDCG@50     MAP   P@10  COMPOSITE")
    print("  " + "-" * (W3 - 2))
    for lbl, sc in [("S  head=15 (base)", sc_s_ref),
                     ("T  head=18",        sc_t),
                     ("U  head=20",        sc_u)]:
        g  = ">" if sc["composite"] > BASELINE_S else "≤"
        p  = "✓" if sc["p10"] >= 1.0 else "✗"
        print(f"  {lbl:<24} {sc['ndcg10']:.4f}  {sc['ndcg50']:.4f}  "
              f"{sc['map']:.4f}  {sc['p10']:.2f}{p}  {sc['composite']:.4f}{g}")
    print("=" * W3)

    valid = [(sc, rws, nm) for sc, rws, nm in
             [(sc_t, rows_t, "T"), (sc_u, rows_u, "U")]
             if sc["composite"] > BASELINE_S and sc["p10"] >= 1.0]

    if valid:
        best_sc, best_rows, best_lbl = max(valid, key=lambda x: x[0]["composite"])
        write_csv(best_rows, "submission_final.csv")
        print(f"  Winner: Variant {best_lbl}  COMPOSITE {best_sc['composite']:.4f} > {BASELINE_S}"
              f"  P@10 {best_sc['p10']:.2f}  → wrote submission_final.csv")
    else:
        print("  Gate NOT cleared — submission_final.csv unchanged.")
        for sc, _, nm in [(sc_t, rows_t, "T"), (sc_u, rows_u, "U")]:
            rs = []
            if sc["composite"] <= BASELINE_S:
                rs.append(f"COMP {sc['composite']:.4f}≤{BASELINE_S}")
            if sc["p10"] < 1.0:
                rs.append(f"P@10 {sc['p10']:.2f}<1.0")
            print(f"    {nm}: {', '.join(rs)}")
    print("=" * W3)

    # ── Controlled head-promotion experiment (Variants V / W) ─────────────────
    pool3 = [scored[i] for i in sorted(range(len(scored)), key=lambda i: -scored[i][4])[:100]]
    pool3_by_order = sorted(pool3, key=_cascade_sort_key)
    S_head_ids = {it[0].get("candidate_id", "") for it in pool3_by_order[:15]}

    def _build_promo_variant(cset_k):
        head_candidates = pool3_by_order[:cset_k]
        ordered_cset    = sorted(head_candidates, key=_t5_order_key)
        head            = ordered_cset[:15]
        head_ids        = {it[0].get("candidate_id", "") for it in head}
        tail            = sorted(
            (it for it in pool3 if it[0].get("candidate_id", "") not in head_ids),
            key=lambda it: -it[2]["pillars"]["domain_evidence"])
        return _make_cascade_rows(head + tail, 100), head

    rows_v, head_v = _build_promo_variant(25)
    rows_w, head_w = _build_promo_variant(20)
    sc_v = _score_against_labels(rows_v, labels_path)
    sc_w = _score_against_labels(rows_w, labels_path)

    s_head_ordered = sorted(pool3_by_order[:15], key=_t5_order_key)
    s_tail = sorted(pool3_by_order[15:], key=lambda it: -it[2]["pillars"]["domain_evidence"])
    rows_s_ref2 = _make_cascade_rows(s_head_ordered + s_tail, 100)
    sc_s_ref2   = _score_against_labels(rows_s_ref2, labels_path)

    W4 = 74
    print("\n" + "=" * W4)
    print("  HEAD-PROMOTION EXPERIMENT — VARIANTS V / W  (base: Variant S 0.7229)")
    print("  Fixed: pool=100 by final_select, tail by domain_evidence")
    print("  Head reorder key: final_order*(0.85+0.15*t5) [same as S]")
    print("  " + "-" * (W4 - 2))
    print(f"  {'Variant':<30} NDCG@10  NDCG@50     MAP   P@10  COMPOSITE")
    print("  " + "-" * (W4 - 2))
    for lbl, sc in [("S  cset=15→15 (base)", sc_s_ref2),
                     ("V  cset=25→top15",     sc_v),
                     ("W  cset=20→top15",     sc_w)]:
        g = ">" if sc["composite"] > BASELINE_S else "≤"
        p = "✓" if sc["p10"] >= 1.0 else "✗"
        print(f"  {lbl:<30} {sc['ndcg10']:.4f}  {sc['ndcg50']:.4f}  "
              f"{sc['map']:.4f}  {sc['p10']:.2f}{p}  {sc['composite']:.4f}{g}")
    print("=" * W4)

    for vname, head_x in [("V", head_v), ("W", head_w)]:
        promoted = [it for it in head_x
                    if it[0].get("candidate_id", "") not in S_head_ids]
        if promoted:
            print(f"\n  Variant {vname} — promoted into head-15 (not in S):")
            for it in promoted:
                cid  = it[0].get("candidate_id", "")
                tier = tier_map.get(cid, "?")
                t5v  = t5_map.get(cid, 0.0)
                fo   = it[1]
                fo_rank = next((i+1 for i, x in enumerate(pool3_by_order)
                                if x[0].get("candidate_id","") == cid), "?")
                print(f"    {cid}  tier={tier}  t5={t5v:.3f}  "
                      f"final_order={fo:.4f}  pool_fo_rank={fo_rank}")
        else:
            print(f"\n  Variant {vname} — no promotions (same 15 as S)")

    valid = [(sc, rws, nm) for sc, rws, nm in
             [(sc_v, rows_v, "V"), (sc_w, rows_w, "W")]
             if sc["composite"] > BASELINE_S and sc["p10"] >= 1.0]

    print()
    if valid:
        best_sc, best_rows, best_lbl = max(valid, key=lambda x: x[0]["composite"])
        write_csv(best_rows, "submission_final.csv")
        print(f"  Winner: Variant {best_lbl}  COMPOSITE {best_sc['composite']:.4f} > {BASELINE_S}"
              f"  P@10 {best_sc['p10']:.2f}  → wrote submission_final.csv")
    else:
        print("  Gate NOT cleared — submission_final.csv unchanged.")
        for sc, _, nm in [(sc_v, rows_v, "V"), (sc_w, rows_w, "W")]:
            rs = []
            if sc["composite"] <= BASELINE_S:
                rs.append(f"COMP {sc['composite']:.4f}≤{BASELINE_S}")
            if sc["p10"] < 1.0:
                rs.append(f"P@10 {sc['p10']:.2f}<1.0")
            print(f"    {nm}: {', '.join(rs)}")
    print("=" * W4)
