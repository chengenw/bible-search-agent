"""
report_utils.py — Aggregation + Markdown helpers shared by run_eval and
leaderboard. Deliberately dependency-free (no chromadb / litellm) so the
leaderboard can summarise saved results without loading the agent stack.

A "question" here is a plain dict {"category": str, "runs": [run, ...]} — the
exact shape stored in evaluation/results/*.json. Each non-refusal run holds
"pool"/"answer" score dicts; each refusal run holds "refusal_pass".
"""

import statistics

# Stable, readable column order for by-category tables; unknown categories sort
# last, alphabetically. "other" is a first-class bucket for future questions.
CATEGORY_ORDER = ["factual", "thematic", "doctrinal", "principle",
                  "scoped", "other", "refusal"]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def cat_sort_key(c: str):
    return (CATEGORY_ORDER.index(c) if c in CATEGORY_ORDER else len(CATEGORY_ORDER), c)


def md_pct(x) -> str:
    return "—" if x is None else f"{x * 100:.0f}%"


def summarize(questions: list[dict]) -> dict:
    """
    Aggregate one model's per-question runs into headline metrics.

    - answer_hit  : end-to-end hit rate — do the verses the agent SHOWS the user
                    contain every required verse? Scored over ALL shown verses
                    (no top-k cutoff), because the user reads all of them.
    - retr@3/@10  : retrieval ranking — is a required verse within the top-k of
                    the agent's merged search pool? (diagnoses search quality.)
    - retr_ceiling: required verse found ANYWHERE in the pool — the upper bound
                    the answer could have reached. The gap to answer_hit is the
                    LLM's selection/formatting loss.
    - refusal     : accuracy on out-of-scope questions (agent correctly declines).
    - by_category : answer_hit per category (refusal accuracy for the refusal bucket).
    """
    scored = [q for q in questions if q["category"] != "refusal"]
    refus = [q for q in questions if q["category"] == "refusal"]

    def field(items, src, key):
        # `if src in r` tolerates no-RAG runs, which have no retrieval "pool".
        return _mean([_mean([r[src][key] for r in q["runs"] if src in r]) for q in items])

    def refusal_field(items):
        return _mean([_mean([r["refusal_pass"] for r in q["runs"]]) for q in items])

    def norag_faith_field(items):
        # None unless these are --no-rag runs (which carry per-citation faithfulness).
        return _mean([_mean([r["norag_faith"]["faithful"] for r in q["runs"]
                             if r.get("norag_faith")]) for q in items])

    cats: dict[str, list] = {}
    for q in questions:
        cats.setdefault(q["category"], []).append(q)
    by_cat = {
        c: (refusal_field(v) if c == "refusal" else field(v, "answer", "ceiling"))
        for c, v in cats.items()
    }

    return {
        "n": len(questions),
        "answer_hit": field(scored, "answer", "ceiling"),
        "retr@3": field(scored, "pool", "pass@3"),
        "retr@10": field(scored, "pool", "pass@10"),
        "retr_ceiling": field(scored, "pool", "ceiling"),
        "refusal": refusal_field(refus) if refus else None,
        "norag_faithful": norag_faith_field(scored),   # None for RAG runs
        "by_category": by_cat,
    }
