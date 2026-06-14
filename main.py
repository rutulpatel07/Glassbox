"""
main.py
=======
CLI runner for the ML Hiring Engine.

Usage Examples
──────────────
# Step 1: Generate 1 lakh candidates
python main.py --generate --count 100000

# Step 2: Run ranking for a job
python main.py --candidates candidates.json --job sample_job.json --top 100

# Step 3: Filter by domain + custom output
python main.py --candidates candidates.json --job sample_job.json \
               --domain tech --top 200 --output top_ml_engineers.json

# Generate a smaller test set quickly
python main.py --generate --count 5000 --candidates test_candidates.json
python main.py --candidates test_candidates.json --job sample_job.json --top 50
"""

import argparse
import json
import os
import sys
import time


DEFAULT_CONFIG = {
    "weights": {
        "skills":            0.30,
        "experience":        0.25,
        "education":         0.15,
        "salary_fit":        0.10,
        "location":          0.08,
        "career_trajectory": 0.07,
        "certifications":    0.03,
        "availability":      0.02
    }
}


def banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║          ML HIRING ENGINE  —  Pure Math, No LLM             ║
║  TF-IDF · Cosine Sim · Gaussian · KMeans · Z-Score · MCDM  ║
╚══════════════════════════════════════════════════════════════╝
""")


def parse_args():
    p = argparse.ArgumentParser(
        description="Rank up to 1 lakh job candidates using pure ML mathematics."
    )
    p.add_argument("--candidates", default="candidates.json",
                   help="Path to candidates JSON  (default: candidates.json)")
    p.add_argument("--job",        default="sample_job.json",
                   help="Path to job requirements JSON  (default: sample_job.json)")
    p.add_argument("--config",     default="config.json",
                   help="Path to weights config JSON  (default: config.json)")
    p.add_argument("--top",        type=int, default=100,
                   help="How many top candidates to output  (default: 100)")
    p.add_argument("--output",     default="results.json",
                   help="Output file for ranked results  (default: results.json)")
    p.add_argument("--domain",     default=None,
                   help="Filter by domain: tech / management / design / finance / marketing")
    p.add_argument("--show",       type=int, default=20,
                   help="How many rows to print in the summary table  (default: 20)")
    p.add_argument("--generate",   action="store_true",
                   help="Generate synthetic candidates before ranking")
    p.add_argument("--count",      type=int, default=100_000,
                   help="Number of candidates to generate  (default: 100000)")
    return p.parse_args()


def load_config(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            cfg = json.load(f)
        print(f"  ✓ Config loaded from {path}")
        return cfg
    print(f"  ℹ  Config file not found ({path}), using defaults.")
    return DEFAULT_CONFIG


def main():
    banner()
    args = parse_args()

    # ── Optional: generate dataset ──────────────────────────────────────────
    if args.generate:
        from generate_candidates import generate_dataset
        generate_dataset(args.count, args.candidates)
        print()

    # ── Validate inputs ──────────────────────────────────────────────────────
    missing = []
    if not os.path.exists(args.candidates):
        missing.append(f"  ✗  Candidates file not found: {args.candidates}")
    if not os.path.exists(args.job):
        missing.append(f"  ✗  Job file not found: {args.job}")

    if missing:
        print("\n".join(missing))
        print("\n  Tip: run with --generate to create a sample candidate dataset.")
        sys.exit(1)

    # ── Load config and engine ──────────────────────────────────────────────
    config = load_config(args.config)

    from ml_engine import HiringMLEngine
    engine = HiringMLEngine(config)

    # ── Load data ───────────────────────────────────────────────────────────
    df  = engine.load_candidates(args.candidates)
    job = engine.load_job(args.job)

    # ── Run ML pipeline ──────────────────────────────────────────────────────
    t0 = time.perf_counter()

    top, full_df, importance = engine.run(
        df, job,
        top_n=args.top,
        filter_domain=args.domain
    )

    elapsed = time.perf_counter() - t0
    rate    = len(df) / elapsed

    print(f"\n  ⏱  {elapsed:.2f}s  |  {rate:,.0f} candidates/sec")

    # ── Display ──────────────────────────────────────────────────────────────
    engine.print_summary(top, show=args.show)

    # ── Export ───────────────────────────────────────────────────────────────
    engine.export_results(top, args.output)

    # ── Additional stats file ─────────────────────────────────────────────
    stats_path = args.output.replace(".json", "_stats.json")
    stats = {
        "job":                  job["title"],
        "total_candidates":     len(df),
        "ranked_candidates":    len(top),
        "processing_time_sec":  round(elapsed, 3),
        "candidates_per_sec":   round(rate, 1),
        "score_stats": {
            "max":    round(float(full_df["composite_score"].max()), 4),
            "mean":   round(float(full_df["composite_score"].mean()), 4),
            "median": round(float(full_df["composite_score"].median()), 4),
            "min":    round(float(full_df["composite_score"].min()), 4),
            "std":    round(float(full_df["composite_score"].std()), 4),
        },
        "tier_distribution": full_df["candidate_tier"].value_counts().to_dict(),
        "anomalies_flagged":  int(full_df["anomaly_flags"].apply(lambda x: len(x) > 0).sum()),
        "feature_importance": importance,
        "weights_used":       config.get("weights", {}),
    }

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  ✓ Stats   written → {stats_path}")
    print(f"\n  Done.  Top {args.top} candidates saved to {args.output}\n")


if __name__ == "__main__":
    main()
