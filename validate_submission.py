#!/usr/bin/env python3
"""
validate_submission.py — Spec-compliance validator for Redrob submission CSVs
=============================================================================
Checks every Section 3 rule:
  1. File is UTF-8 encoded
  2. Exactly 1 header row + 100 data rows (101 lines total)
  3. Header columns exactly: candidate_id, rank, score, reasoning (in order)
  4. Ranks 1-100 each appear exactly once
  5. candidate_ids are unique
  6. All candidate_ids are present in the candidate pool (candidates.jsonl)
  7. Score is monotone non-increasing with rank
  8. No empty (or whitespace-only) reasoning cells

Usage:
  python validate_submission.py submission.csv
  python validate_submission.py submission.csv --candidates /path/to/candidates.jsonl

Exit code: 0 = all pass, 1 = one or more failures.
"""

import argparse
import csv
import json
import os
import sys

EXPECTED_COLS  = ["candidate_id", "rank", "score", "reasoning"]
EXPECTED_NROWS = 100
EXPECTED_RANKS = set(range(1, EXPECTED_NROWS + 1))

# Default pool path (same directory as this script)
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_POOL   = os.path.join(_HERE, "candidates.jsonl")


# ── Reporting helper ──────────────────────────────────────────────────────────

_results: list[tuple[bool, str, str]] = []


def _check(label: str, passed: bool, detail: str = "") -> bool:
    _results.append((passed, label, detail))
    status = "PASS" if passed else "FAIL"
    line   = f"  [{status}] {label}"
    if detail:
        line += f"\n           {detail}"
    print(line)
    return passed


# ── Individual checks ─────────────────────────────────────────────────────────

def check_utf8(path: str):
    try:
        with open(path, encoding="utf-8", errors="strict") as f:
            content = f.read()
        _check("UTF-8 encoding", True)
        return content
    except UnicodeDecodeError as exc:
        _check("UTF-8 encoding", False, str(exc))
        return None


def check_parse(content: str):
    """Return list of rows (including header) or None on parse error."""
    try:
        rows = list(csv.reader(content.splitlines()))
        _check("CSV parseable", True)
        return rows
    except Exception as exc:
        _check("CSV parseable", False, str(exc))
        return None


def check_header(rows: list[list[str]]) -> bool:
    header = rows[0] if rows else []
    ok = header == EXPECTED_COLS
    detail = f"Got: {header}" if not ok else ""
    return _check(
        "Header: candidate_id, rank, score, reasoning",
        ok,
        detail,
    )


def check_row_count(data_rows: list[list[str]]) -> bool:
    n  = len(data_rows)
    ok = n == EXPECTED_NROWS
    return _check(
        f"Exactly {EXPECTED_NROWS} data rows",
        ok,
        f"Found {n} rows" if not ok else "",
    )


def parse_data(data_rows: list[list[str]]) -> list[tuple]:
    """
    Parse each data row into (candidate_id, rank_int, score_float, reasoning).
    Returns a list; entries are None for rows that fail parsing.
    """
    parsed = []
    bad = []
    for i, row in enumerate(data_rows, start=2):  # 1-indexed; row 1 = header
        if len(row) < 4:
            bad.append(f"line {i}: too few columns ({len(row)})")
            parsed.append(None)
            continue
        cid, rank_s, score_s, reasoning = row[0], row[1], row[2], row[3]
        try:
            rank  = int(rank_s)
            score = float(score_s)
        except ValueError as exc:
            bad.append(f"line {i}: {exc}")
            parsed.append(None)
            continue
        parsed.append((cid, rank, score, reasoning))
    if bad:
        _check("All rows have valid numeric rank & score", False, "; ".join(bad[:5]))
    else:
        _check("All rows have valid numeric rank & score", True)
    return parsed


def check_ranks(valid: list[tuple]) -> bool:
    ranks     = [r[1] for r in valid]
    rank_set  = set(ranks)
    dup_ranks = sorted({r for r in ranks if ranks.count(r) > 1})
    missing   = sorted(EXPECTED_RANKS - rank_set)
    extra     = sorted(rank_set - EXPECTED_RANKS)
    ok = (rank_set == EXPECTED_RANKS) and (len(ranks) == EXPECTED_NROWS)
    detail = ""
    if not ok:
        parts = []
        if missing:  parts.append(f"missing: {missing}")
        if extra:    parts.append(f"out-of-range: {extra}")
        if dup_ranks: parts.append(f"duplicated: {dup_ranks}")
        detail = "; ".join(parts)
    return _check("Ranks 1-100 each exactly once", ok, detail)


def check_unique_ids(valid: list[tuple]) -> bool:
    cids = [r[0] for r in valid]
    dups = sorted({c for c in cids if cids.count(c) > 1})
    ok   = len(dups) == 0
    return _check(
        "candidate_ids unique",
        ok,
        f"Duplicated IDs: {dups}" if not ok else "",
    )


def check_ids_in_pool(valid: list[tuple], pool_path: str) -> bool:
    submission_ids = {r[0] for r in valid}

    if not os.path.isfile(pool_path):
        return _check(
            "All candidate_ids in candidates.jsonl",
            False,
            f"Pool file not found: {pool_path}",
        )

    print(f"  [....] Loading pool from {os.path.basename(pool_path)} …",
          end="\r", flush=True)
    pool_ids: set[str] = set()
    with open(pool_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                cid = json.loads(line).get("candidate_id")
                if cid:
                    pool_ids.add(cid)
            except json.JSONDecodeError:
                pass

    missing = sorted(submission_ids - pool_ids)
    ok      = len(missing) == 0
    return _check(
        "All candidate_ids in candidates.jsonl",
        ok,
        (f"Not found in pool ({len(pool_ids):,} total): {missing}"
         if not ok
         else f"pool size: {len(pool_ids):,}"),
    )


def check_monotone_score(valid: list[tuple]) -> bool:
    """Score must be non-increasing when rows are sorted by rank."""
    by_rank = sorted(valid, key=lambda r: r[1])
    violations = []
    for i in range(1, len(by_rank)):
        prev_rank, prev_score = by_rank[i - 1][1], by_rank[i - 1][2]
        curr_rank, curr_score = by_rank[i][1],     by_rank[i][2]
        if curr_score > prev_score + 1e-9:  # float-safe epsilon
            violations.append(
                f"rank {curr_rank} score={curr_score:.6f} > "
                f"rank {prev_rank} score={prev_score:.6f}"
            )
    ok = len(violations) == 0
    detail = "; ".join(violations[:3])
    if len(violations) > 3:
        detail += f" … (+{len(violations) - 3} more)"
    return _check("Score monotone non-increasing with rank", ok, detail)


def check_no_empty_reasoning(valid: list[tuple]) -> bool:
    empty = sorted(r[1] for r in valid if not r[3].strip())
    ok    = len(empty) == 0
    return _check(
        "No empty reasoning cells",
        ok,
        f"Empty at ranks: {empty}" if not ok else "",
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Validate a Redrob submission CSV against the spec."
    )
    ap.add_argument("submission", help="Path to submission CSV")
    ap.add_argument(
        "--candidates",
        default=DEFAULT_POOL,
        help=f"Candidate pool JSONL (default: {DEFAULT_POOL})",
    )
    args = ap.parse_args()

    print(f"\nRedrob submission validator")
    print(f"  File      : {args.submission}")
    print(f"  Pool      : {args.candidates}")
    print("=" * 60)

    # 1. UTF-8
    content = check_utf8(args.submission)
    if content is None:
        _print_summary()
        sys.exit(1)

    # 2. Parse CSV
    rows = check_parse(content)
    if rows is None:
        _print_summary()
        sys.exit(1)

    # 3. Header
    check_header(rows)
    data_rows = rows[1:]

    # 4. Row count
    check_row_count(data_rows)

    # 5. Parse individual cells
    parsed = parse_data(data_rows)
    valid  = [r for r in parsed if r is not None]

    # 6–10. Per-record checks (only meaningful if we have rows)
    if valid:
        check_ranks(valid)
        check_unique_ids(valid)
        check_ids_in_pool(valid, args.candidates)
        check_monotone_score(valid)
        check_no_empty_reasoning(valid)

        # Summary stats
        scores = [r[2] for r in sorted(valid, key=lambda r: r[1])]
        print(f"\n  Score range : {min(scores):.6f} – {max(scores):.6f}")
        rl = [len(r[3]) for r in valid]
        print(f"  Reasoning   : min={min(rl)} max={max(rl)} avg={sum(rl)//len(rl)} chars")

    _print_summary()
    n_fail = sum(1 for ok, _, _ in _results if not ok)
    sys.exit(0 if n_fail == 0 else 1)


def _print_summary():
    n_pass = sum(1 for ok, _, _ in _results if ok)
    n_fail = sum(1 for ok, _, _ in _results if not ok)
    print("=" * 60)
    if n_fail == 0:
        print(f"  RESULT: ALL {n_pass} CHECKS PASSED ✓")
    else:
        print(f"  RESULT: {n_fail} FAILED / {n_pass} PASSED")
    print()


if __name__ == "__main__":
    main()
