"""
leaderboard.py — Combine every run in evaluation/results/ into Markdown
comparison tables, one row per model. This is the resume/README artifact:
"benchmarked N models on a 100-question eval set; here's how they rank."

    python -m evaluation.leaderboard            # latest run per model
    python -m evaluation.leaderboard --all      # every saved run (dated rows)
    python -m evaluation.leaderboard --version KJV   # filter by translation

Reads the saved JSON directly — no agent/LLM/vector store is loaded. A --no-rag
baseline result (see run_eval) is folded into a "No-RAG" column beside its RAG
twin; the Answer hit − No-RAG gap is the measured retrieval benefit.
"""

import argparse
import json
from pathlib import Path

from evaluation.report_utils import cat_sort_key, md_pct, summarize

RESULTS_DIR = Path(__file__).parent / "results"


def _load(files):
    for f in files:
        try:
            yield f, json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue


def _dataset(p) -> str:
    """Dataset stem from the payload's csv field ('verses' for legacy files)."""
    return Path(p.get("csv") or "verses.csv").stem


def _mode(p) -> str:
    """'no_rag' for baseline files, 'rag' otherwise (legacy files have no mode)."""
    return p.get("mode") or "rag"


def _latest_per_model(pairs):
    """Keep the newest result (by timestamp) per (model, version, dataset, mode)."""
    best: dict[tuple, tuple] = {}
    for f, p in pairs:
        key = (p.get("model", "?"), p.get("version", "?"), _dataset(p), _mode(p))
        stamp = p.get("timestamp", "") or f.name
        if key not in best or stamp > best[key][0]:
            best[key] = (stamp, f, p)
    return [(f, p) for _, f, p in best.values()]


def _table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate eval results into a leaderboard.")
    ap.add_argument("--all", action="store_true",
                    help="one row per run (default: newest run per model)")
    ap.add_argument("--version", help="only include results for this translation")
    ap.add_argument("--dataset", help="only include results for this dataset "
                                      "(CSV stem, e.g. 'verses' or 'data')")
    ap.add_argument("--models", help="comma-separated recorded model names to include "
                                     "(e.g. 'mimo-v2.5,deepseek-v4-flash'); "
                                     "eval_models.sh passes its roster here")
    args = ap.parse_args()

    pairs = list(_load(sorted(RESULTS_DIR.glob("*.json"))))
    if args.version:
        v = args.version.upper()
        pairs = [(f, p) for f, p in pairs if p.get("version", "").upper() == v]
    if args.dataset:
        pairs = [(f, p) for f, p in pairs if _dataset(p) == args.dataset]
    if args.models:
        want = {m.strip() for m in args.models.split(",") if m.strip()}
        pairs = [(f, p) for f, p in pairs if p.get("model") in want]
    if not pairs:
        print(f"No result files in {RESULTS_DIR}. Run `python -m evaluation.run_eval` first.")
        return
    if not args.all:
        pairs = _latest_per_model(pairs)

    # No-RAG baselines are folded into an extra column beside their RAG twin
    # (keyed by model, version, dataset) rather than shown as their own rows.
    # A no_rag file with NO rag twin still gets a row of its own so it isn't
    # lost. At --all, pairing one-to-one is ambiguous, so all rows show as-is.
    norag, rag_keys = {}, set()
    if not args.all:
        for f, p in pairs:
            key = (p.get("model"), p.get("version"), _dataset(p))
            if _mode(p) == "no_rag":
                sm = summarize(p["questions"])
                # keep n (to flag mismatched coverage) and faithfulness
                norag[key] = (sm["answer_hit"], sm["n"], sm.get("norag_faithful"))
            else:
                rag_keys.add(key)
        pairs = [(f, p) for f, p in pairs
                 if _mode(p) != "no_rag"
                 or (p.get("model"), p.get("version"), _dataset(p)) not in rag_keys]

    rows = []
    for f, p in pairs:
        s = summarize(p["questions"])
        base = p.get("model", "?") + (" [no-RAG]" if _mode(p) == "no_rag" else "")
        s["label"] = base if not args.all else f"{base} ({p.get('timestamp','')})"
        s["version"] = p.get("version", "?")
        s["dataset"] = _dataset(p)
        s["mode"] = _mode(p)
        s["runs"] = p.get("runs", "?")
        s["norag"] = norag.get((p.get("model"), p.get("version"), _dataset(p)))
        rows.append(s)
    rows.sort(key=lambda s: (s["answer_hit"] is not None, s["answer_hit"] or 0), reverse=True)

    # union of categories across models, in canonical order
    cats = sorted({c for s in rows for c in s["by_category"]}, key=cat_sort_key)

    # Different datasets aren't comparable rows; show which is which, but only
    # once a second dataset actually exists (keeps the common case unchanged).
    show_ds = len({s["dataset"] for s in rows}) > 1
    show_norag = any(s["norag"] is not None for s in rows)
    show_faith = any(s["norag"] and s["norag"][2] is not None for s in rows)

    def norag_cell(s):
        # No-RAG hit; append its own n when it was scored over fewer/other
        # questions than the RAG run, so an uneven comparison can't mislead.
        if not s["norag"]:
            return "—"
        hit, n, _faith = s["norag"]
        return md_pct(hit) + (f" (n={n})" if n != s["n"] else "")

    def faith_cell(s):
        # Fraction of the no-RAG model's own citations whose text matches the
        # real verse — RAG is faithful by construction, so this column is
        # no-RAG only; the low value is its hallucination rate's complement.
        return md_pct(s["norag"][2]) if s["norag"] else "—"

    print(f"\n# Bible search agent — model leaderboard  ({len(rows)} model(s))\n")
    print("**Answer hit** = end-to-end: verses the agent shows the user contain the gold "
          "(all shown verses scored, no cutoff).  "
          "**Retrieval@k** = a required verse is within top-k of the agent's search.")
    if show_norag:
        print("**No-RAG** = same questions answered from the model's own knowledge "
              "(no retrieval); the Answer hit − No-RAG gap is the measured RAG benefit.")
    if show_faith:
        print("**No-RAG faithful** = fraction of the no-RAG model's own citations whose "
              "quoted text matches the real verse (RAG is verbatim, so ~100%); the "
              "shortfall is its hallucination rate.")
    print()

    headline = _table(
        ["Model", "Ver"] + (["Data"] if show_ds else [])
        + ["Runs", "n", "Answer hit"]
        + (["No-RAG"] if show_norag else []) + (["No-RAG faithful"] if show_faith else [])
        + ["Retrieval@3", "Retrieval@10", "Refusal"],
        [[s["label"], str(s["version"])] + ([s["dataset"]] if show_ds else [])
         + [str(s["runs"]), str(s["n"]), md_pct(s["answer_hit"])]
         + ([norag_cell(s)] if show_norag else []) + ([faith_cell(s)] if show_faith else [])
         + [md_pct(s["retr@3"]), md_pct(s["retr@10"]), md_pct(s["refusal"])]
         for s in rows],
    )
    print(headline)

    print("\n## By category (answer hit rate)\n")
    percat = _table(
        ["Model"] + cats,
        [[s["label"] + (f" ({s['dataset']})" if show_ds else "")]
         + [md_pct(s["by_category"].get(c)) for c in cats] for s in rows],
    )
    print(percat)


if __name__ == "__main__":
    main()
