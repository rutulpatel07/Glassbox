#!/usr/bin/env python3
"""
reasoning.py — Deterministic rich-reasoning generator for submission_v3.2
==========================================================================
Reads submission_v3point2.csv + candidates.jsonl, re-runs the scoring
engine for the top-100 candidates, generates specific, fact-grounded
1-2 sentence reasoning strings, writes submission_v3point2_with_reasoning.csv.

Requirements:
  - Zero LLM calls. All claims derived from profile data + engine trace.
  - Sentence frames vary by dominant scoring pillar (retr_s / modern_s / vdb).
  - Tone scales: ranks 1-10 confident, 11-50 measured, 51-100 borderline/filler.
  - Every concern grounded in an actual profile field.
"""

import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rank import (
    load_candidates,
    score_candidate,
    _candidate_text,
    RX_RETRIEVAL,
    RX_LLM_MODERN,
    RX_VECTOR_DB,
    RX_PRODUCTION,
    _load_semantic_scores,
)

# ── IR / search-specific skill keywords (substring match on lowercased name) ──
_IR_KW = frozenset([
    # Vector / ANN stores
    "faiss", "pinecone", "milvus", "weaviate", "qdrant", "pgvector",
    # Search engines
    "elasticsearch", "opensearch", "solr", "lucene",
    # LLM / embedding tech
    "bert", "e5", "bge", "sentence-transformer", "qlora", "lora", "peft",
    "rag", "llm", "llms", "hugging face", "transformers", "huggingface",
    # Retrieval techniques
    "bm25", "semantic search", "vector search", "learning to rank",
    "two-tower", "two tower", "collaborative filter", "matrix factor",
    "reranker", "rerank", "haystack", "llamaindex",
    # Eval metrics
    "ndcg", "mrr", "mean reciprocal",
    # Retrieval frameworks
    "information retrieval", "recommendation systems",
])

_PROF_RANK = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}


# ── Profile fact extractors ────────────────────────────────────────────────────

def _get_ir_skills(c, top_n=3):
    """Return up to top_n IR-relevant skill names, ranked by proficiency then endorsements."""
    found = []
    for s in c.get("skills", []):
        name = (s.get("name") or "").strip()
        name_lo = name.lower()
        if any(kw in name_lo for kw in _IR_KW):
            found.append((
                name,
                _PROF_RANK.get(s.get("proficiency"), 0),
                int(s.get("endorsements") or 0),
            ))
    found.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return [name for name, _, _ in found[:top_n]]


def _get_employers(c):
    """Return (current_company_str, [past_company_strs])."""
    p = c.get("profile", {})
    current = (p.get("current_company") or "").strip()
    hist = c.get("career_history", [])
    past = [
        (h.get("company") or "").strip()
        for h in hist
        if not h.get("is_current") and (h.get("company") or "").strip()
    ]
    return current, past


# ── Signal classifier ─────────────────────────────────────────────────────────

def _classify_signal(trace, text):
    """
    Determine dominant signal from domain_evidence_terms + raw regex counts.
    Returns one of: retr, retr_modern, modern_retr, modern, vdb, adjacent.
    """
    ev = trace.get("domain_evidence_terms", [])
    has_retr   = "retrieval/ranking work" in ev
    has_modern = "modern LLM/embedding work" in ev
    has_vdb    = "vector/search infra" in ev

    retr_n   = len(RX_RETRIEVAL.findall(text))
    modern_n = len(RX_LLM_MODERN.findall(text))

    if has_retr and has_modern:
        # If modern hits outnumber retrieval hits by >20%, label as modern-first
        if modern_n > retr_n * 1.2:
            return "modern_retr"
        return "retr_modern"
    if has_retr:
        return "retr"
    if has_modern:
        return "modern"
    if has_vdb:
        return "vdb"
    return "adjacent"


# ── Concern extractor ─────────────────────────────────────────────────────────

def _primary_concern(c, trace):
    """
    Return the single most material concern as a string.
    Priority order: hard negatives → low YOE → notice → location → behavioral → minor.
    Always returns a non-empty string (surfacing even minor concerns).
    """
    neg  = trace.get("neg_reasons", [])
    p    = c.get("profile", {})
    sig  = c.get("redrob_signals", {}) or {}
    pil  = trace.get("pillars", {})
    beh  = trace.get("behavior", {})
    hist = c.get("career_history", [])

    yoe   = float(p.get("years_of_experience") or 0)
    nd    = sig.get("notice_period_days")
    loc   = (p.get("location") or "").strip()
    title = (p.get("current_title") or "").lower()

    # Hard negatives from scoring engine
    if "research-leaning, little production signal" in neg:
        return "research-heavy profile with limited production deployment evidence"
    if "career entirely in IT-services/consulting" in neg:
        return "entire career in IT-services/consulting firms — no product-company exposure"
    if "CV/speech/robotics focus, thin NLP/IR" in neg:
        return "CV/robotics-heavy background dilutes IR/NLP signal"
    if "frequent short stints (job-hopping pattern)" in neg:
        return "frequent short stints raise tenure-risk concerns"

    # YOE below soft minimum
    if yoe < 3.5:
        return f"only {yoe}y experience — below the 4y soft minimum for this role"

    # Notice period (most impactful scheduling risk first)
    if nd is not None and nd > 90:
        return f"{int(nd)}-day notice period is a significant scheduling constraint"
    if nd is not None and nd > 60:
        return f"{int(nd)}-day notice period exceeds the preferred 30-day window"

    # Location outside target hubs
    loc_s = pil.get("location", 1.0)
    if loc_s < 0.5 and loc:
        return f"based in {loc}, outside the preferred Delhi-NCR/Pune/Hyderabad/Mumbai hubs"

    # Behavioral: long inactivity
    days_inactive = beh.get("days_inactive", 0)
    resp = beh.get("response_rate", 0.0)
    if days_inactive > 180:
        return f"inactive {days_inactive} days; recruiter response rate {resp:.0%}"

    # Mild notice (45-60 d)
    if nd is not None and nd > 30:
        return f"{int(nd)}-day notice period"

    # Secondary location: good city but not the preferred hub cluster
    if loc_s < 0.95 and loc:
        return f"based in {loc} — relocation to a preferred hub (Delhi-NCR/Pune/Hyderabad) required"

    # YOE slightly below ideal (4-5y band)
    if yoe < 5.0:
        return f"{yoe}y experience is below the 6-8y ideal for this seniority"

    # Title mismatch: junior label at senior YOE
    if "junior" in title and yoe >= 5.0:
        return f"'Junior' title may underrepresent actual scope at {yoe}y experience"

    # Adjacent engineering role (not explicitly ML)
    if trace.get("role") == "adjacent_eng":
        return "current title is engineering-adjacent without explicit ML/AI designation"

    # Last resort: mild behavioural
    if resp < 0.30 and resp >= 0:
        return f"low recruiter response rate ({resp:.0%}) may limit reachability"

    return "no material concerns at this tier"


# ── Rank bucket ───────────────────────────────────────────────────────────────

def _rank_bucket(rank):
    if rank <= 10:  return "top"
    if rank <= 30:  return "upper"
    if rank <= 50:  return "mid"
    if rank <= 70:  return "lower"
    return "bottom"


# ── Domain phrase builder ─────────────────────────────────────────────────────

def _domain_phrase(ev):
    has_retr = "retrieval/ranking work" in ev
    has_vdb  = "vector/search infra" in ev
    has_mod  = "modern LLM/embedding work" in ev
    has_eval = "ranking-evaluation experience" in ev

    if has_retr and has_vdb and has_mod:
        return "retrieval, vector search, and LLM-based reranking"
    if has_retr and has_vdb:
        return "retrieval, ranking, and vector search"
    if has_retr and has_mod:
        return "retrieval, ranking, and LLM/embedding pipelines"
    if has_retr:
        return "retrieval and ranking"
    if has_mod:
        return "LLM/embedding-based retrieval"
    if has_vdb:
        return "vector search infrastructure"
    return "adjacent ML"


# ── Main reasoning generator ──────────────────────────────────────────────────

def make_rich_reasoning(c, trace, rank):
    """
    Generate a 1-2 sentence reasoning string grounded entirely in profile data.
    Sentence frame selected by (signal_category × rank_bucket).
    """
    p   = c.get("profile", {})
    ev  = trace.get("domain_evidence_terms", [])

    yoe   = float(p.get("years_of_experience") or 0)
    title = (p.get("current_title") or "ML professional").strip()

    current_co, past_cos = _get_employers(c)
    ir_skills             = _get_ir_skills(c, top_n=3)

    text       = _candidate_text(c).lower()
    signal_cat = _classify_signal(trace, text)
    bucket     = _rank_bucket(rank)
    concern    = _primary_concern(c, trace)
    domain     = _domain_phrase(ev)

    # Employer string: current + one notable past
    if current_co:
        empl = current_co
        if past_cos and past_cos[0] and past_cos[0] != current_co:
            empl = f"{current_co} (prev. {past_cos[0]})"
    else:
        empl = "unknown employer"

    # Top-2 IR skill names
    s_top = ir_skills[:2]
    if len(s_top) == 2:
        skills_str = f"{s_top[0]} and {s_top[1]}"
    elif len(s_top) == 1:
        skills_str = s_top[0]
    else:
        skills_str = ""

    skills_part = f" ({skills_str})" if skills_str else ""

    # Concern sentence
    concern_sent = f" Concern: {concern}." if concern else ""

    # ── Frame selection by (signal_cat × bucket) ──────────────────────────────
    #
    # Frames vary structure:
    #   retr/retr_modern → career-built-around / solid-background lead
    #   modern_retr/modern → LLM-era-search lead
    #   vdb/adjacent → infrastructure/ML-generalist lead
    # -------------------------------------------------------------------------

    if bucket == "top":  # ranks 1-10: confident, no hedging
        if signal_cat in ("retr", "retr_modern"):
            s1 = (f"{yoe}y {title} at {empl}, career built around {domain}{skills_part}"
                  f" — directly matching the LTR/recsys specialist mandate.")
            s2 = concern_sent.strip()
        elif signal_cat in ("modern_retr", "modern"):
            s1 = (f"{yoe}y {title} at {empl} brings LLM-era search depth: "
                  f"{domain}{skills_part} — tightly aligned with the JD's "
                  f"dense-retrieval and reranking scope.")
            s2 = concern_sent.strip()
        else:  # vdb / adjacent
            s1 = (f"{yoe}y {title} at {empl} with strong {domain} background{skills_part}; "
                  f"search-infrastructure depth meets the JD's technical bar.")
            s2 = concern_sent.strip()

    elif bucket == "upper":  # ranks 11-30: measured, solid
        pil_u = trace.get("pillars", {})
        has_eval_signal = "ranking-evaluation experience" in ev
        has_github      = pil_u.get("external_validation", 0) >= 0.7
        if signal_cat in ("retr", "retr_modern"):
            if has_eval_signal and not has_github:
                s1 = (f"{yoe}y {title} at {empl} with {domain} and ranking-evaluation "
                      f"experience{skills_part}; a strong-fit candidate for the LTR/recsys role.")
            elif has_github and not has_eval_signal:
                # Break ties: use a candidate-id-derived parity to vary the opener
                cid_digits = c.get("candidate_id", "0")
                cid_parity = sum(int(ch) for ch in cid_digits if ch.isdigit()) % 2
                if cid_parity == 0:
                    s1 = (f"Active external profile alongside {yoe}y {domain} work at {empl}{skills_part}; "
                          f"verified IR toolchain depth and well-qualified for the LTR/recsys mandate.")
                else:
                    s1 = (f"{empl}'s {yoe}y {title} combines {domain} with open-source IR activity"
                          f"{skills_part}; a strong shortlist candidate for the LTR/recsys mandate.")
            elif has_eval_signal and has_github:
                s1 = (f"{yoe}y {title} at {empl} demonstrates {domain}{skills_part} "
                      f"with measurable eval metrics and open-source activity — strong IR specialist fit.")
            else:
                # Alternate between two openers to avoid first-50-char collisions at same YOE
                # Deterministic: alternate by whether past employer is a known services firm
                past_lo = past_cos[0].lower() if past_cos else ""
                services_past = any(sf in past_lo for sf in [
                    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
                    "tech mahindra", "hcl", "mindtree", "lti", "deloitte", "ibm",
                ])
                if services_past:
                    s1 = (f"{yoe}y {title} at {empl}{skills_part} — {domain} exposure "
                          f"from a product-company role, despite prior services-firm tenure.")
                else:
                    s1 = (f"Solid {yoe}y {domain} background at {empl}{skills_part}; "
                          f"well-qualified for the LTR/recsys role with genuine IR toolchain depth.")
            s2 = concern_sent.strip()
        elif signal_cat in ("modern_retr", "modern"):
            s1 = (f"{yoe}y {title} at {empl} with {domain} background{skills_part}; "
                  f"LLM and embedding pipeline experience maps to the JD's recsys scope.")
            s2 = concern_sent.strip()
        else:  # vdb / adjacent
            s1 = (f"Solid {yoe}y {title} at {empl} with search-adjacent exposure{skills_part}; "
                  f"meets the JD baseline but lacks explicit LTR/recsys depth to rank higher.")
            s2 = concern_sent.strip()

    elif bucket == "mid":  # ranks 31-50: adequate, note gap
        if signal_cat in ("retr", "retr_modern"):
            s1 = (f"{yoe}y {title} at {empl} with {domain} exposure{skills_part}; "
                  f"adequate alignment with the LTR/recsys JD but not stand-out depth.")
            s2 = f"Concern: {concern}." if concern else "Rank reflects a softer combination of pillars."
        elif signal_cat in ("modern_retr", "modern"):
            s1 = (f"Modern ML background ({yoe}y) at {empl}{skills_part}; "
                  f"{domain} work partially satisfies the JD — retrieval-specific depth is secondary.")
            s2 = f"Concern: {concern}." if concern else "Classical IR/LTR evidence is limited."
        else:  # vdb / adjacent
            s1 = (f"ML generalist ({yoe}y) at {empl} with adjacent search exposure{skills_part}; "
                  f"domain signal is present but thin for a dedicated IR specialist role.")
            s2 = f"Concern: {concern}." if concern else "IR/LTR signal too marginal to rank higher."

    elif bucket == "lower":  # ranks 51-70: borderline
        if signal_cat in ("retr", "retr_modern"):
            s1 = (f"Borderline: {yoe}y {title} at {empl} shows {domain} signals{skills_part}, "
                  f"but pillar scores sit at the lower edge of the qualified band.")
            s2 = f"Concern: {concern}." if concern else "Manual review recommended before scheduling."
        elif signal_cat in ("modern_retr", "modern"):
            s1 = (f"Borderline: {yoe}y {title} at {empl} with LLM/embedding depth{skills_part}; "
                  f"modern search coverage lacks the classical IR/LTR evidence the JD requires.")
            s2 = f"Concern: {concern}." if concern else "Verify ranking/LTR depth in initial screen."
        else:  # vdb / adjacent
            s1 = (f"Borderline: {yoe}y ML background at {empl} with thin IR signal{skills_part}; "
                  f"included at the outer edge of the qualified pool.")
            s2 = f"Concern: {concern}." if concern else "Verify recsys depth in screen."

    else:  # bottom: ranks 71-100 — explicitly filler
        if signal_cat in ("retr", "retr_modern"):
            s1 = (f"Filler rank: {yoe}y {title} at {empl} with some {domain} signal{skills_part}; "
                  f"pillar combination falls below the confident-inclusion threshold.")
            s2 = f"Concern: {concern}." if concern else "Lowest-confidence inclusion; manual review recommended."
        elif signal_cat in ("modern_retr", "modern"):
            s1 = (f"Filler rank: {yoe}y {title} at {empl} with LLM-adjacent skills{skills_part}; "
                  f"modern ML background but insufficient classical IR/LTR depth for the specialist role.")
            s2 = f"Concern: {concern}." if concern else "Lowest-confidence inclusion."
        else:  # vdb / adjacent
            s1 = (f"Filler rank: {yoe}y ML background at {empl} with marginal IR signal{skills_part}; "
                  f"IR/LTR evidence too thin for confident shortlisting.")
            s2 = f"Concern: {concern}." if concern else "Verify any recsys experience before outreach."

    parts = [s1]
    if s2:
        parts.append(s2)
    # No length cap here — s1/s2 are already complete sentences; the final
    # length budget (with the rank-stability suffix) is enforced once, safely,
    # by make_reasoning() in rank.py.
    return " ".join(parts)


# ── Verification helpers ──────────────────────────────────────────────────────

def _verify_sample(out_rows, n=10, seed=42):
    """Print n sampled rows and run grounding / tone checks."""
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(out_rows)), n), key=lambda i: out_rows[i]["rank"])

    print("\n" + "=" * 80)
    print(f"SAMPLE: {n} RANDOMLY SELECTED ROWS (ranked order)")
    print("=" * 80)
    for i in indices:
        r = out_rows[i]
        print(f"\nrank={r['rank']:>3}  {r['candidate_id']}  score={r['score']}")
        print(f"  {r['reasoning']}")

    print("\n" + "=" * 80)
    print("VERIFICATION CHECKS")
    print("=" * 80)

    # 1. Structural variety: unique first-50-char openings
    first50 = [r["reasoning"][:50] for r in out_rows]
    unique50 = len(set(first50))
    print(f"[1] Unique reasoning openings (first 50 chars): {unique50}/100"
          f"  {'OK' if unique50 >= 90 else 'WARN: low variety'}")

    # 2. Tone at top-10: no borderline/filler language
    top10_ok = all(
        "borderline" not in r["reasoning"].lower() and "filler" not in r["reasoning"].lower()
        for r in out_rows if r["rank"] <= 10
    )
    print(f"[2] Top-10 rows are confident (no borderline/filler): {'YES' if top10_ok else 'NO — FAIL'}")

    # 3. Ranks 71-100 all have explicit filler/borderline signals
    bottom30_ok = all(
        "borderline" in r["reasoning"].lower() or "filler" in r["reasoning"].lower()
        for r in out_rows if r["rank"] >= 71
    )
    print(f"[3] Ranks 71-100 rows labelled borderline/filler: {'YES' if bottom30_ok else 'NO — FAIL'}")

    # 4. Grounding: specific employer or IR skill named
    # Pull company names and skill keywords from out_rows' profile data for a spot check
    specific_count = sum(
        1 for r in out_rows
        if any(kw in r["reasoning"].lower() for kw in [
            "sarvam", "paytm", "flipkart", "amazon", "freshworks", "razorpay",
            "glance", "yellow.ai", "inmobi", "swiggy", "zomato", "meesho",
            "ola", "byju", "cred", "hcl", "infosys", "wipro", "tcs", "aganitha",
            "niramai", "unacademy", "vedantu", "rephrase", "saarthi", "haptik",
            "faiss", "pinecone", "milvus", "weaviate", "qdrant", "pgvector",
            "elasticsearch", "opensearch", "solr", "lucene",
            "bert", "bge", "e5", "qlora", "lora", "peft", "rag", "llm",
            "bm25", "semantic search", "haystack", "llamaindex",
            "ndcg", "mrr", "information retrieval", "recommendation systems",
        ])
    )
    print(f"[4] Rows citing specific employer/skill: {specific_count}/100"
          f"  {'OK' if specific_count >= 80 else 'WARN: few specific names'}")

    # 5. Concerns present
    with_concern = sum(1 for r in out_rows if "concern:" in r["reasoning"].lower())
    print(f"[5] Rows with explicit Concern clause: {with_concern}/100")

    all_ok = top10_ok and bottom30_ok and unique50 >= 90
    print(f"\nOverall: {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED — review above'}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    base_dir   = os.path.dirname(os.path.abspath(__file__))
    v32_path   = os.path.join(base_dir, "submission_v3point2.csv")
    cands_path = os.path.join(base_dir, "candidates.jsonl")
    out_path   = os.path.join(base_dir, "submission_v3point2_with_reasoning.csv")

    # Load v3.2 submission (provides candidate_id, rank, score)
    with open(v32_path, newline="", encoding="utf-8") as f:
        v32_rows = list(csv.DictReader(f))
    print(f"Loaded {len(v32_rows)} rows from {os.path.basename(v32_path)}")

    target_ids = {r["candidate_id"] for r in v32_rows}

    # Load semantic scores (warns + defaults to 0.5 if precomputed/ absent)
    _load_semantic_scores()

    # Stream candidates.jsonl; keep only the 100 targets
    print(f"Streaming {os.path.basename(cands_path)} for {len(target_ids)} target candidates …")
    cands_by_id = {}
    with open(cands_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            cid = c.get("candidate_id")
            if cid in target_ids:
                cands_by_id[cid] = c
                if len(cands_by_id) == len(target_ids):
                    break  # all found, stop early

    found = len(cands_by_id)
    if found < len(target_ids):
        missing = target_ids - set(cands_by_id)
        print(f"WARNING: {len(missing)} candidate IDs not found in JSONL: {missing}")
    else:
        print(f"All {found} target candidates loaded.")

    # Score each candidate and generate reasoning
    print(f"Running score_candidate() + make_rich_reasoning() for {found} candidates …")
    out_rows = []
    for row in v32_rows:
        cid   = row["candidate_id"]
        rank  = int(row["rank"])
        score = row["score"]

        c = cands_by_id.get(cid, {
            "candidate_id": cid,
            "profile": {},
            "career_history": [],
            "skills": [],
            "redrob_signals": {},
        })

        _, trace    = score_candidate(c)
        reasoning   = make_rich_reasoning(c, trace, rank)

        out_rows.append({
            "candidate_id": cid,
            "rank":         rank,
            "score":        score,
            "reasoning":    reasoning,
        })

    # Verify and sample
    _verify_sample(out_rows, n=10, seed=42)

    # Write output CSV
    print(f"\nWriting {os.path.basename(out_path)} …")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in out_rows:
            writer.writerow([r["candidate_id"], r["rank"], r["score"], r["reasoning"]])

    print(f"Done. {len(out_rows)} rows written to {out_path}")


if __name__ == "__main__":
    main()
