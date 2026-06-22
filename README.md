# Redrob — Intelligent Candidate Discovery & Ranking

A **glass-box, fully deterministic, zero-AI, zero-network** candidate ranker for the
Redrob hackathon. The ranking and the reasoning are the *same object*: every score is
a transparent decomposition of named factors, and the reasoning column is a
deterministic readout of that trace — so it can never contradict the rank or
hallucinate a fact the candidate doesn't have.

## Reproduce the submission (single command)

```bash
pip install -r requirements.txt
python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv
```

Runs CPU-only, no GPU, no network, no LLM, no model weights — well under the
5 min / 16 GB budget on the full 100k pool. Output is spec-compliant:
`candidate_id,rank,score,reasoning`, exactly 100 rows, unique ranks 1–100,
monotonic non-increasing score, deterministic tie-break (`candidate_id` ascending).

## How it ranks (pipeline)

| Layer | What it does |
|---|---|
| L0 Honeypot gate | Internal-consistency checks (expert skill w/ 0 months, skill duration > career, tenure vs. dates, `is_current` contradictions) → **hard floor**. Defends the >10% honeypot DQ. |
| L1 Role-fit gate | Non-engineering titles (HR/Marketing/Content/…) floored regardless of skills → defeats keyword-stuffers. |
| L2 Domain evidence | Reads `career_history.description`, not skill names. **Explicit** retrieval/ranking/recsys/vector/eval signal scores high; **adjacent** plain-language ML/data signal is hard-capped below explicit fits (catches Tier-5s without inflating filler). |
| L3 Fit pillars | Seniority (6–8y), product-vs-services, skill substance (proficiency×tenure×endorsement×assessment), Python, eval-frameworks, GitHub/external (−1 = neutral), location, notice. |
| L4 do-NOT-want | Research-only, consulting-only, CV/speech/robotics-only, job-hopping → penalty (not floor). |
| L5 Behavioral | Availability multiplier from last-active recency, response rate, interview completion, open-to-work, completeness. |

## Validation harness (run before submitting — no live leaderboard)

```bash
python validation_harness.py make-labels --candidates sample_candidates.json --out labels_template.csv
#   fill the 'tier' column 0..5 by hand (you + Rutul), then:
python validation_harness.py score    --candidates sample_candidates.json --labels labels_filled.csv
python validation_harness.py ablate   --candidates sample_candidates.json   # weight-stability
python validation_harness.py dual     --candidates sample_candidates.json   # two-scorer agreement
python validation_harness.py selftest --candidates sample_candidates.json   # injects traps, checks flooring
```

## Files
- `rank.py` — the engine (single-command CSV producer).
- `validation_harness.py` — NDCG/MAP scorer, ablation, dual-scorer agreement, adversarial self-test.
- `requirements.txt`, `submission_metadata.yaml`.
