# How the Ranking Engine Works — for Rutul

*Written for someone who's a strong engineer but new to ML. No jargon without explanation.*

---

## 1. The problem in one line

We're given **100,000 candidate profiles** and one **job description (JD)**. We must output the **top 100 candidates**, best-fit first, each with a one-line reason. A scoring system we can't see grades our list.

The catch: the organizers **deliberately planted traps** to punish the obvious approach (keyword matching). So the whole game is *not* matching keywords — it's reasoning like a good recruiter would.

---

## 2. The single most important idea

> **We did NOT ask an AI "who's the best candidate?"** We built a transparent recruiter's scorecard in plain Python, and every number it produces can be explained.

This matters because the hackathon has 5 judging stages, and stages 3–5 are humans checking whether you can *reproduce and defend* your work. A team that let an AI pick the ranking can't explain it. We can explain every single placement, because **the ranking and the reasons come from the same calculation**. More on this in section 6.

There is **no AI, no machine-learning model, no internet** inside the engine. It's pure arithmetic and text pattern-matching. That's a deliberate choice — it makes the result identical every time you run it (judges need that) and fully explainable.

---

## 3. How a single candidate gets scored (the assembly line)

Think of each candidate passing through 6 stations. Each station either adjusts a score or raises a red flag.

### Station 0 — "Is this profile even real?" (Honeypot gate)
Some planted profiles are **logically impossible** — e.g. "8 years at a company, but the dates only add up to 3 years," or "expert in 10 skills they've used for 0 months." A human recruiter would smell these instantly.

We check internal consistency: do the dates match the claimed durations? Did they "use a skill longer than they've worked"? Are they "currently employed" in two places at once? If a profile fails these, we **crush its score to the bottom**. 

*Why it matters:* if more than 10% of our top 100 are these fakes, we're **disqualified**. So this gate is non-negotiable.

### Station 1 — "Is this the right kind of role?" (Role gate)
The nastiest trap: an **HR Manager** or **Marketing Manager** whose profile is stuffed with AI keywords (RAG, LLM, embeddings…). Keyword matching ranks them #1. A recruiter knows an HR Manager isn't an AI Engineer no matter what words are listed.

So we look at the **job title** first. Non-engineering titles get floored regardless of their skill list. This single rule defeats the keyword-stuffer trap.

### Station 2 — "Did they actually BUILD the right things?" (Evidence reading)
This is the cleverest part. The JD wants people who built **search / recommendation / ranking / retrieval** systems. But the best candidates often **don't use those buzzwords** — they write "built the system that suggests products to users" instead of "recommendation engine."

So instead of matching skill *names*, we **read the free-text descriptions** of what they did at each job. We scan for two layers:
- **Explicit signals** — clear retrieval/ranking/search/vector-database work. Scores high.
- **Plain-language signals** — adjacent clues like "feature pipelines," "worked with the data science team," "fine-tuned a model." Catches the hidden-gem candidates — but **capped**, so a vaguely-relevant person never outranks someone with clear, direct experience.

*This is the "reads between the lines like a recruiter" part.*

### Station 3 — The fit checklist
Straightforward recruiter checks, each a number from 0 to 1:
- **Seniority** — JD wants ~6–8 years. A bell curve peaks there; too junior or too senior scores lower.
- **Product vs. services company** — JD prefers people from product companies over pure IT-services/consulting (TCS/Infosys/etc.). Having *some* product experience cancels the penalty.
- **Skill substance** — not "do they list Python?" but "how good, how long, how endorsed, what did they score on the platform's skill test?"
- **Python / evaluation-knowledge / GitHub activity / location / notice period** — smaller checks. (Note: "no GitHub" is treated as *neutral*, not bad — many great engineers don't use it.)

### Station 4 — The red-flag deductions (do-NOT-want list)
The JD explicitly lists things to avoid: pure-academic-research backgrounds, careers spent entirely at consulting firms, people who only did computer-vision/robotics (different field), and job-hoppers. These **subtract** from the score (a penalty, not a total floor — they could still be okay).

### Station 5 — "Are they actually reachable?" (Behavioral multiplier)
A perfect-on-paper candidate who **hasn't logged in for 6 months and ignores 95% of recruiter messages** is useless for hiring. The JD says to treat behavior as a *multiplier*. So we take the score from stations 0–4 and **multiply** it by an "availability" factor built from: last-active date, recruiter response rate, interview show-up rate, open-to-work flag.

---

## 4. The final number

```
final score = (checklist total)         ← stations 2 & 3
            × (role gate)               ← station 1
            × (1 − red-flag penalty)    ← station 4
            × (availability)            ← station 5
            then crushed to bottom if honeypot ← station 0
```

Sort everyone by this number, take the top 100. Ties broken by candidate ID so the result is always identical.

---

## 5. How it was built (the engineering)

- **Language:** plain Python with `numpy` (just for math). One file, `rank.py`.
- **Reading the data:** the 100K candidates come as a `.jsonl` file (one JSON profile per line). We stream it line-by-line so it never blows up memory, even at 465 MB.
- **The "reading between the lines":** Python **regular expressions** (regex) — patterns that find phrases like "recommendation" or "feature pipeline" in text. We compile ~10 pattern groups once and scan each candidate's text.
- **Speed:** scoring 100K profiles is ~90 seconds (limit is 5 minutes). Two tricks got us there: lowercasing text once instead of case-insensitive matching (1.8× faster), and a hand-written date parser (6× faster than Python's built-in). The slow part is the text scanning — that's ~70% of the time.

---

## 6. Why our reasoning column can't lie (the differentiator)

Most teams will generate the ranking one way and then ask an AI to *write reasons* separately. Those reasons often **contradict the rank** or **invent facts** — and the judges check for exactly that.

Ours is different: the reason is **printed directly from the same numbers** that produced the rank. If the score says "weak domain evidence," the reason literally reads that off. It **cannot** praise a candidate the score buried, and it **cannot** mention a skill the candidate doesn't have, because it only ever reads from that candidate's actual computed values. This is our edge in the human-judged rounds.

---

## 7. How we know it works (before submitting blind)

There's no live scoreboard — we get graded once. So we built a **validation harness** to prove the engine works first:
- **Self-test:** we inject fake keyword-stuffers and honeypots into the pool and confirm the engine buries them. (It does — they land ~40,000th and ~90,000th.)
- **Dual-scorer:** a second, independently-written scorer ranks the candidates too; we trust the top candidates both agree on. (They agreed on 7 of the top 10 on real data; the 3 disagreements are flagged for us to eyeball.)
- **Ablation:** we wiggle every weight up and down 40% and check the top 10 barely changes. (80% stayed put — meaning the ranking reads real signal, not a lucky knob setting.)
- **Tuning:** once you and I hand-label ~50 real candidates, `tune.py` fits the tier cutoffs to our judgment instead of guesses.

---

## 8. What's still on us

The engine is tuned against the JD and a sample that was mostly filler (no genuine star candidates to calibrate on). **The one thing left is for us to hand-label ~40–60 real candidates together** so the tier boundaries reflect reality. That's the difference between "good" and "as good as it gets." Everything else is built, tested, and fast.
