"""
Evaluate the agent against a labelled CSV of questions (default verses.csv; any
CSV with the same columns works via --csv).

Each question runs --runs times and is scored all-or-nothing on two verse lists:
  • final answer   — verses the agent shows the user (end-to-end; every shown
    verse counts, no top-k cutoff).
  • retrieval pool — every retrieved verse, deduped and cosine-ranked; scored
    at top-3 and top-10 to measure how highly the search ranks the gold.

Gold columns, all non-empty ones required for a pass:
    Answer all       every entry present
    Answer any       ≥1 entry
    Answer any two   ≥2 distinct entries
    Answer any plus  ≥1 entry; side B of a contested question (both sides needed)
A range like "Pro 6:16-19" is one entry, satisfied by any verse inside it.
"refusal" questions have empty gold and pass when the agent declines.

Results are cached per (translation, dataset, model) under results/ and re-scored
from raw output on every run, so gold/parser fixes apply without new LLM calls;
--reeval forces fresh answers.

Usage:
    python -m evaluation.run_eval                    # KJV, 3 runs, verses.csv
    python -m evaluation.run_eval --runs 1 --limit 1 # quick smoke test
    python -m evaluation.run_eval --no-rag           # memory-only baseline
    ./evaluation/eval_models.sh --runs 1             # several models + leaderboard
"""

import argparse
import csv
import io
import json
import logging
import random
import re
import time
from collections import OrderedDict
from contextlib import redirect_stdout
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import litellm

# Silence litellm's background WARNINGs (e.g. "Unmapped finish_reason") that
# would otherwise interleave with our progress output.
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

from src.config import MODEL, REASONING_EFFORT, CACHE_BASE
from src.vector_store import get_collection
from src.bible_loader import load_all_verses
from src.agent import run_agent, _LLM_KWARGS

from evaluation.refs import (
    ReferenceError,
    format_ref,
    lookup_book,
    parse_cell,
    parse_reference,
)
from evaluation.report_utils import _mean, cat_sort_key, md_pct, summarize

KS = (3, 10)                          # top-k cutoffs for the retrieval hit rate
RESULTS_DIR = Path(__file__).parent / "results"

# A scripture citation in any format the models produce ("[John 3:16]" or bare
# "1. John 3:16 — …"): optional book number, book word (+ "of X" for Song of
# Songs), chapter:verse, optional range. Book validity is checked via lookup_book.
_CITE_RE = re.compile(
    r"((?:[1-3]\s+)?[A-Z][A-Za-z]+(?:\s+[Oo]f\s+[A-Z][A-Za-z]+)?)[.\s]+(\d+):(\d+)(?:\s*[-–]\s*(\d+))?"
)

# Bare model name for results files: drop the litellm provider prefix and
# OpenRouter routing suffixes (:free, :nitro, …) so all variants of a model
# share one file. Size tags like ollama's ":3b" are kept.
MODEL_NAME = re.sub(r":(free|beta|extended|nitro|floor|online)$",
                    "", MODEL.split("/")[-1])

# Transient provider errors worth a backoff retry; anything else fails immediately.
_RETRYABLE = tuple(
    exc for exc in (
        getattr(litellm, name, None)
        for name in ("RateLimitError", "Timeout", "APIConnectionError",
                     "ServiceUnavailableError", "InternalServerError")
    ) if isinstance(exc, type)
)


# Gold labels

class Gold:
    """Parsed gold labels for one question."""

    def __init__(self, question: str, category: str, answer_all: str,
                 answer_any: str, answer_any_two: str = "",
                 answer_any_plus: str = "", comment: str = ""):
        self.question = question
        self.category = (category or "uncategorized").strip().lower()
        self.all_entries = parse_cell(answer_all)        # every entry must be hit
        # "any" groups as (label, entries, min_hits): each non-empty group must
        # have at least min_hits DISTINCT entries hit. "Answer any" needs 1,
        # "Answer any two" needs 2, "Answer any plus" needs 1 and serves as side
        # B of a contested question (a pass requires a verse from BOTH sides).
        # The label is kept for the "missed: ..." diagnostics in the run log.
        self.any_groups = [
            (label, grp, need) for label, grp, need in (
                ("Answer any",      parse_cell(answer_any), 1),
                ("Answer any two",  parse_cell(answer_any_two), 2),
                ("Answer any plus", parse_cell(answer_any_plus), 1),
            ) if grp
        ]
        self.raw_all = answer_all.strip()
        self.raw_any = answer_any.strip()
        self.raw_any_two = answer_any_two.strip()
        self.raw_any_plus = answer_any_plus.strip()
        self.comment = (comment or "").strip()

    @property
    def is_refusal(self) -> bool:
        """Out-of-scope question whose correct answer is the agent's refusal."""
        return self.category == "refusal"

    @property
    def gold_tuples(self) -> set:
        """Union of every verse tuple across all columns (for MRR)."""
        out = set()
        for entry in self.all_entries:
            out.update(entry)
        for _label, grp, _need in self.any_groups:
            for entry in grp:
                out.update(entry)
        return out


def _field(row: dict, *names: str) -> str:
    """First non-empty value among the given column names (current + legacy)."""
    for n in names:
        v = row.get(n)
        if v and v.strip():
            return v.strip()
    return ""


def load_gold(csv_path: Path) -> list[Gold]:
    # utf-8-sig also accepts a BOM from Excel; anything not valid UTF-8 (e.g. a
    # dash pasted from a GBK/Chinese-IME document) fails loudly with the offset.
    try:
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    except UnicodeDecodeError as e:
        raise SystemExit(
            f"❌  {csv_path} is not valid UTF-8 (byte {e.object[e.start]:#04x} at "
            f"offset {e.start}) — re-save the file as UTF-8 and use plain '-' in ranges.")
    return [
        Gold(
            row["Question"].strip(),
            _field(row, "Category"),
            _field(row, "Answer all"),
            _field(row, "Answer any"),
            _field(row, "Answer any two"),
            _field(row, "Answer any plus", "Answer any2"),   # "Answer any2" = legacy name
            _field(row, "Comment", "comments"),
        )
        for row in rows
    ]


def validate_gold(gold: list[Gold], version: str) -> None:
    """Warn about gold references that do not exist in the chosen translation."""
    verses = load_all_verses(version)
    # Resolve each stored book name through lookup_book, NOT a strict `in
    # BOOK_ID` test — some translations label books differently (Bolls' ESV uses
    # "Psalm"/"Song of Solomon" vs the canonical "Psalms"/"Song of Songs"). They
    # still score correctly because scoring also resolves via lookup_book; using
    # the strict test here falsely flagged every Psalms reference as missing.
    valid = set()
    for v in verses:
        try:
            valid.add((lookup_book(v["book"]), v["chapter"], v["verse"]))
        except ReferenceError:
            continue
    entries = lambda g: g.all_entries + [e for _l, grp, _n in g.any_groups for e in grp]
    problems = []
    for g in gold:
        for entry in entries(g):
            missing = [t for t in entry if t not in valid]
            if missing and len(missing) == len(entry):
                problems.append((g.question, format_ref(entry[0]), len(entry)))
    if problems:
        print(f"\n⚠️   {len(problems)} gold reference(s) do not exist in {version}:")
        for q, ref, n in problems:
            span = f" (range of {n})" if n > 1 else ""
            print(f"      • {ref}{span}  —  '{q}'")
        print("      These entries can never be satisfied; fix the CSV.\n")

    # Label-structure sanity: a non-refusal question with no gold at all would
    # trivially score 100%, and an "any" group with fewer entries than its
    # required minimum can never be satisfied.
    for g in gold:
        if g.is_refusal:
            continue
        if not g.all_entries and not g.any_groups:
            print(f"⚠️   no gold labels (would always pass) — '{g.question}'")
        for label, grp, need in g.any_groups:
            if len(grp) < need:
                print(f"⚠️   '{label}' requires {need} hits but lists only "
                      f"{len(grp)} entr{'y' if len(grp)==1 else 'ies'} — '{g.question}'")


# Candidate verse lists from one agent run

def retrieval_pool(trace: list[dict]) -> list[tuple]:
    """Merge all retrieval results into one score-ranked, deduped tuple list."""
    best: dict[tuple, float] = {}
    for call in trace:
        for r in call["results"]:
            try:
                tup = parse_reference(r["reference"])[0]
            except ReferenceError:
                continue
            score = r.get("score", 0.0)
            if tup not in best or score > best[tup]:
                best[tup] = score
    return [t for t, _ in sorted(best.items(), key=lambda kv: kv[1], reverse=True)]


def cited_verses(answer: str) -> list[tuple]:
    """
    Verses the agent SHOWS the user, in order of appearance, deduped.

    Scans the answer for references in any format (bracketed or bare) and keeps
    only those whose book resolves — so bracketed prose ("[him]") and stray
    capitalised words are ignored. This is what the user actually sees, NOT the
    larger pool the tool returned for the model to choose from.
    """
    out, seen = [], set()
    for m in _CITE_RE.finditer(answer):
        book, chap, v1, v2 = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        try:
            book_id = lookup_book(book)
        except ReferenceError:
            continue
        end = int(v2) if v2 is not None else v1
        if end < v1:
            continue
        for v in range(v1, end + 1):
            t = (book_id, chap, v)
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out


def is_refusal(answer: str) -> bool:
    """
    True when the agent declined to answer from scripture — the correct response
    to an out-of-scope question. The retriever always returns nearest neighbours,
    so refusal is an LLM decision visible only in the final text: it either emits
    the canonical "no relevant verses" line or cites no verses at all.
    """
    return "no relevant verses" in answer.lower() or not cited_verses(answer)


# Scoring

def _entry_hit(entry, candset) -> bool:
    return any(t in candset for t in entry)


def question_pass(g: Gold, candset: set) -> bool:
    """Pass = every 'answer all' entry hit AND each 'any' group reaches its
    minimum number of DISTINCT entries hit (1; 2 for 'Answer any two')."""
    all_ok = all(_entry_hit(e, candset) for e in g.all_entries)
    groups_ok = all(sum(_entry_hit(e, candset) for e in grp) >= need
                    for _label, grp, need in g.any_groups)
    return all_ok and groups_ok


def _fmt_entry(entry) -> str:
    """One gold entry for display — 'Book c:v', with '-v2' when it's a range."""
    s = format_ref(entry[0])
    if len(entry) > 1:
        s += f"-{entry[-1][2]}"
    return s


def missed_requirements(g: Gold, candset: set) -> list[str]:
    """
    Which gold requirements the shown verses failed — one line each, for the
    ❌ run log (so a failure is diagnosable without opening the results file).
    """
    out = []
    missing = [_fmt_entry(e) for e in g.all_entries if not _entry_hit(e, candset)]
    if missing:
        out.append(f"'Answer all' still needs: {', '.join(missing)}")
    for label, grp, need in g.any_groups:
        hit = sum(_entry_hit(e, candset) for e in grp)
        if hit < need:
            rest = [_fmt_entry(e) for e in grp if not _entry_hit(e, candset)]
            shown = ", ".join(rest[:6]) + (" …" if len(rest) > 6 else "")
            out.append(f"'{label}' needs {need} hit(s), got {hit} — candidates: {shown}")
    return out


def graded_recall_all(g: Gold, candset: set):
    """Fraction of 'answer all' entries hit (None if no answer-all column)."""
    if not g.all_entries:
        return None
    return sum(_entry_hit(e, candset) for e in g.all_entries) / len(g.all_entries)


def mrr(g: Gold, cand: list[tuple]) -> float:
    gold = g.gold_tuples
    for i, t in enumerate(cand, start=1):
        if t in gold:
            return 1.0 / i
    return 0.0


def score_candidate(g: Gold, cand: list[tuple]) -> dict:
    """Metrics for one candidate list against one question."""
    out = {"mrr": mrr(g, cand), "ceiling": float(question_pass(g, set(cand)))}
    for k in KS:
        cset = set(cand[:k])
        out[f"pass@{k}"] = float(question_pass(g, cset))
        out[f"recall_all@{k}"] = graded_recall_all(g, cset)
    return out


def faithfulness(answer: str, pool: list[tuple]):
    """Fraction of cited verses that appear in the retrieval pool (None if none cited)."""
    cited = cited_verses(answer)
    if not cited:
        return None
    poolset = set(pool)
    return sum(t in poolset for t in cited) / len(cited)


# No-RAG faithfulness (hallucination check). RAG copies retrieved text verbatim,
# but the --no-rag baseline recites from memory and can fabricate, so we verify
# each shown verse against the real text by word containment (not exact match —
# models quote partial verses and other translations). The quote is checked
# against every cached translation and the best match wins, so a real verse
# recited in NIV/ESV wording isn't flagged just because we scored against KJV.
FAITH_THRESHOLD = 0.6            # min word-containment for a quote to be "faithful"

_VERSE_TEXT: dict[str, dict] = {}


def verse_text(version: str) -> dict:
    """Cached {(book_id, chapter, verse): text} for a translation."""
    if version not in _VERSE_TEXT:
        table = {}
        for v in load_all_verses(version):
            try:
                table[(lookup_book(v["book"]), v["chapter"], v["verse"])] = v["text"]
            except ReferenceError:
                continue
        _VERSE_TEXT[version] = table
    return _VERSE_TEXT[version]


def cached_versions() -> list[str]:
    """Translations already downloaded to the pickle cache (no network needed)."""
    d = Path(CACHE_BASE)
    return sorted(p.stem for p in d.glob("*.pkl")) if d.exists() else []


def _words(s: str) -> list[str]:
    """Lowercase word list; drop KJV brackets and punctuation so quotes across
    translations/punctuation compare fairly."""
    s = s.lower().replace("[", "").replace("]", "")
    return re.sub(r"[^a-z0-9 ]", " ", s).split()


def _containment(quote: list[str], actual: list[str]) -> float:
    """Fraction of the quote's words that appear, in order, in the actual verse."""
    if not quote:
        return 0.0
    sm = SequenceMatcher(None, quote, actual, autojunk=False)
    return sum(b.size for b in sm.get_matching_blocks()) / len(quote)


# Verse-list entry: a line whose FIRST token (after an optional "1." / "-" / "["
# / bold marker) is a reference. Matching only the leading token avoids counting
# references mentioned in prose; we also stop at the Summary section entirely.
_SUMMARY_RE = re.compile(r"(?im)^\s*\**\s*summary\b")
_LIST_LEAD = re.compile(r"^[\s>*_#•.\-]*(?:\d+[.)])?[\s*_\[]*")


def norag_faithfulness(answer: str, version: str):
    """
    Check each verse a no-RAG answer SHOWS ("N. [Ref] — quoted text") against the
    real verse across EVERY cached translation. A citation is faithful iff its
    reference EXISTS (real verse in some translation) AND most of the quoted
    words appear in that verse in AT LEAST ONE translation — so a correct verse
    recited in any wording passes; only an invented reference or text matching
    no translation fails. Returns fractions {n_cited, ref_valid, faithful}, or
    None if nothing cited. (`version` is included even if not yet cached.)
    """
    tables = [verse_text(v) for v in dict.fromkeys(cached_versions() + [version])]
    body = _SUMMARY_RE.split(answer)[0]        # the verse LIST makes the claims, not the summary
    rows = []
    for line in body.splitlines():
        entry = _LIST_LEAD.sub("", line, count=1)
        m = _CITE_RE.match(entry)              # reference must LEAD the entry
        if not m:
            continue
        ch, v1 = int(m.group(2)), int(m.group(3))
        v2 = int(m.group(4)) if m.group(4) else v1
        try:                                   # unknown book / reversed range → invalid reference
            tuples = [(lookup_book(m.group(1)), ch, v) for v in range(v1, v2 + 1)] if v2 >= v1 else []
        except ReferenceError:
            tuples = []
        ref_ok = any(t in tbl for tbl in tables for t in tuples)
        quote = _words(entry[m.end():])
        best = max((_containment(quote, _words(" ".join(tbl.get(t, "") for t in tuples)))
                    for tbl in tables), default=0.0)
        rows.append((ref_ok, ref_ok and best >= FAITH_THRESHOLD))
    if not rows:
        return None
    n = len(rows)
    return {"n_cited": n,
            "ref_valid": sum(a for a, _ in rows) / n,
            "faithful":  sum(b for _, b in rows) / n}


# Aggregation helpers

def _fmt(x, pct=True):
    if x is None:
        return "  —  "
    return f"{x*100:5.1f}%" if pct else f"{x:5.3f}"


def _agg(records, source, key):
    """Mean over questions of (mean over runs) of records[*]['runs'][*][source][key].
    `if source in run` tolerates no-RAG runs, which carry no retrieval 'pool'."""
    return _mean([_mean([run[source][key] for run in rec["runs"] if source in run])
                  for rec in records])


def _refusal_acc(records):
    return _mean([_mean([r["refusal_pass"] for r in rec["runs"]]) for rec in records])


# Results file (one stable file per version/dataset/model; doubles as answer cache)

def _result_path(version: str, csv_file: str, no_rag: bool = False) -> Path:
    """
    One results file per (version, dataset, mode, model). The dataset must be
    part of the key: saving prunes questions absent from the current CSV, so two
    CSVs sharing one file would silently erase each other's stored answers.
    verses.csv keeps the historical name (no stem) so existing caches stay valid.
    """
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", MODEL_NAME)   # e.g. drop ":" in ollama tags
    stem = Path(csv_file).stem
    dataset = "" if stem == "verses" else re.sub(r"[^A-Za-z0-9._-]+", "_", stem) + "-"
    mode = "norag-" if no_rag else ""
    return RESULTS_DIR / f"{version}-{dataset}{mode}{safe_model}.json"


def load_cache(path: Path) -> dict[str, dict]:
    """Previously stored results, keyed by question text. {} if none/corrupt."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return {q["question"]: q for q in payload.get("questions", [])}


def load_sibling_cache(version: str, csv_file: str, no_rag: bool) -> dict[str, dict]:
    """
    Answers stored under OTHER datasets for the same (version, model, mode).

    Stored runs are raw model output keyed by question text — the dataset is
    irrelevant, the question is what matters. So a question shared between two
    CSVs is answered once and reused everywhere (it is re-scored under the
    current CSV's gold on load, and saved into this dataset's own file).
    """
    primary = _result_path(version, csv_file, no_rag)
    out: dict[str, dict] = {}
    for f in sorted(RESULTS_DIR.glob(f"{version}-*.json")):
        if f == primary:
            continue
        try:
            payload = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if (payload.get("model") != MODEL_NAME
                or payload.get("version") != version
                or (payload.get("mode") == "no_rag") != no_rag):
            continue
        for q in payload.get("questions", []):
            # keep the sibling with the most stored runs for each question
            if q.get("runs") and len(q["runs"]) > len(out.get(q["question"], {}).get("runs") or []):
                out[q["question"]] = q
    return out


def rescore_run(g: Gold, run: dict, no_rag: bool = False, version: str = "") -> dict:
    """
    Re-derive a cached run's scores from its raw fields (answer_text, pool_refs)
    so stored answers are always judged by the CURRENT gold labels and parsing
    code — an old run never carries stale scores forward. Runs from legacy files
    without raw fields are returned unchanged (can't be re-scored).
    """
    answer = run.get("answer_text")
    if answer is None:
        return run
    if g.is_refusal:
        return {"n_calls": run.get("n_calls"), "answer_text": answer,
                "refusal_pass": float(is_refusal(answer))}
    if no_rag:
        # No retrieval pool — score the answer and check citations for fabrication.
        return {
            "n_calls":     run.get("n_calls"),
            "answer_text": answer,
            "answer":      score_candidate(g, cited_verses(answer)),
            "norag_faith": norag_faithfulness(answer, version) if version
                           else run.get("norag_faith"),
        }
    pool = []
    for ref in run.get("pool_refs", []):
        try:
            pool.append(parse_reference(ref)[0])
        except ReferenceError:
            continue
    return {
        "n_calls":     run.get("n_calls"),
        "answer_text": answer,
        "pool_refs":   run.get("pool_refs", []),
        "pool":        score_candidate(g, pool),
        "answer":      score_candidate(g, cited_verses(answer)),
        "faithful":    faithfulness(answer, pool),
        "pool_top":    [format_ref(t) for t in pool[:3]],
    }


# Main

# The --no-rag baseline: same task, same output format, but NO retrieval tool —
# the model answers purely from its own memorized knowledge of the Bible. The
# gap between this and the agent's answer hit rate is the measured RAG benefit
# (and, when no-RAG citations don't exist or are wrong, the hallucination risk).
NO_RAG_PROMPT = """You are a Bible study assistant. Answer from your own knowledge of the Bible.

Format:

**Relevant Verses:**
1. [Reference] — [verse text]
...

**Summary:**
[Synthesis of the verses listed above.]

If the question cannot be answered from the Bible, say exactly:
"No relevant verses were found for this query. Try rephrasing your question."
"""


def run_no_rag(question: str):
    """One no-retrieval baseline answer; returns (answer, empty trace)."""
    resp = litellm.completion(
        model=MODEL,
        messages=[{"role": "system", "content": NO_RAG_PROMPT},
                  {"role": "user",   "content": question}],
        **_LLM_KWARGS,
    )
    return (resp.choices[0].message.content or ""), []


def _call_agent(question: str, col, quiet: bool, retries: int = 2, no_rag: bool = False):
    """
    One agent (or --no-rag) run, retrying transient errors with 5s/10s backoff.
    This is a coarse outer retry (the whole question restarts); the agent already
    retries each completion call, so this is kept short to avoid dozens of attempts.
    """
    for attempt in range(retries + 1):
        try:
            ctx = redirect_stdout(io.StringIO()) if quiet else _nullctx()
            with ctx:
                if no_rag:
                    return run_no_rag(question)
                return run_agent(question, col, return_trace=True)
        except _RETRYABLE as e:
            if attempt == retries:
                raise
            delay = min(60, 5 * 2 ** attempt)
            print(f"      ⏳  {type(e).__name__} — retry {attempt + 1}/{retries} in {delay}s")
            time.sleep(delay)


def order_questions(gold: list[Gold], shuffle: bool, seed: int,
                    stratified: bool) -> list[Gold]:
    """
    Reorder the question set before ``--limit`` is applied.

    The CSV is grouped by category, so a plain ``--limit 10`` only ever sees the
    first category. ``--shuffle`` randomises (reproducibly via ``--seed``);
    ``--stratified`` round-robins across categories so a small limit still spans
    every category. Combined, questions are shuffled within each category first.
    """
    items = list(gold)
    if shuffle:
        random.Random(seed).shuffle(items)
    if stratified:
        buckets: "OrderedDict[str, list]" = OrderedDict()
        for g in items:
            buckets.setdefault(g.category, []).append(g)
        lists = list(buckets.values())
        items, i = [], 0
        while any(i < len(lst) for lst in lists):
            for lst in lists:
                if i < len(lst):
                    items.append(lst[i])
            i += 1
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the Bible search agent.")
    ap.add_argument("--version", default="KJV")
    ap.add_argument("--csv", default="verses.csv")
    ap.add_argument("--runs", type=int, default=3, help="runs per question (averaged)")
    ap.add_argument("--limit", type=int, default=0, help="only first N questions (0=all)")
    ap.add_argument("--shuffle", action="store_true",
                    help="shuffle questions before --limit (reproducible via --seed)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for --shuffle")
    ap.add_argument("--stratified", action=argparse.BooleanOptionalAction, default=True,
                    help="round-robin across categories so a small --limit spans them "
                         "(default on; use --no-stratified for raw CSV order)")
    ap.add_argument("--reeval", action="store_true",
                    help="re-answer questions even if already stored in the results file")
    ap.add_argument("--no-rag", dest="no_rag", action="store_true",
                    help="baseline: answer from the model's own knowledge, no retrieval "
                         "(measures the RAG benefit and hallucination risk)")
    ap.add_argument("--show-agent", action="store_true", help="don't suppress agent output")
    ap.add_argument("--detailed", action="store_true",
                    help="also print ranking diagnostics (MRR, ceiling, recall, faithfulness)")
    args = ap.parse_args()
    version = args.version.upper()

    gold_csv = load_gold(Path(args.csv))
    gold = order_questions(gold_csv, args.shuffle, args.seed, args.stratified)
    if args.limit:
        gold = gold[: args.limit]

    n_comments = sum(1 for g in gold if g.comment)
    mode_tag = "  [NO-RAG baseline]" if args.no_rag else ""
    print(f"\n📊  Evaluating {MODEL_NAME} on {version}{mode_tag}  |  {len(gold)} questions × {args.runs} run(s)")
    if REASONING_EFFORT:
        print(f"    (reasoning effort: {REASONING_EFFORT})")
    if n_comments:
        print(f"    ({n_comments} questions carry a Comment in the CSV)")

    result_path = _result_path(version, args.csv, args.no_rag)
    cache = {} if args.reeval else load_cache(result_path)
    # Reuse the same question answered under a different dataset — answers are
    # keyed by question text, so the dataset is irrelevant (only same model,
    # version and mode). Fills in questions this dataset shares with another CSV.
    if not args.reeval:
        sibling = load_sibling_cache(version, args.csv, args.no_rag)
        n_reused = 0
        for g in gold:
            if not (cache.get(g.question) or {}).get("runs") and sibling.get(g.question):
                cache[g.question] = sibling[g.question]
                n_reused += 1
        if n_reused:
            print(f"    ♻️   {n_reused} question(s) reused from another dataset's results")

    def n_runs_cached(g):
        return len((cache.get(g.question) or {}).get("runs") or [])

    n_full = sum(1 for g in gold if n_runs_cached(g) >= args.runs)
    n_part = sum(1 for g in gold if 0 < n_runs_cached(g) < args.runs)
    if n_full or n_part:
        parts = []
        if n_full:
            parts.append(f"{n_full} fully cached")
        if n_part:
            parts.append(f"{n_part} topping up to {args.runs} run(s)")
        print(f"    ↩️   {' + '.join(parts)} from {result_path.name} (--reeval to re-answer)")
    print()
    validate_gold(gold, version)

    # The vector store (and its embedding model) is only needed for fresh
    # RAG runs — the no-RAG baseline never retrieves.
    col = get_collection(version) if (n_full < len(gold) and not args.no_rag) else None

    records = []
    t0 = time.time()
    consec_failed = 0        # questions in a row where every fresh attempt errored
    try:
        for qi, g in enumerate(gold, start=1):
            print(f"[{qi}/{len(gold)}] {g.question}")

            # Cached runs are reused (re-scored under current gold/parser); if the
            # file holds fewer runs than --runs, only the missing ones are executed.
            cached_runs = [rescore_run(g, r, args.no_rag, version)
                           for r in (cache.get(g.question) or {}).get("runs") or []]
            n_fresh = max(0, args.runs - len(cached_runs))
            if cached_runs:
                if g.is_refusal:
                    v = _mean([r.get("refusal_pass") for r in cached_runs])
                else:
                    v = _mean([r.get("answer", {}).get("ceiling") for r in cached_runs])
                tag = "✅" if v == 1 else ("❌" if v == 0 else "➖")
                more = f", running {n_fresh} more" if n_fresh else ""
                print(f"      ↩️   {tag} cached ({len(cached_runs)} run(s){more})")
                if v == 0 and not g.is_refusal:
                    cs = set(cited_verses(cached_runs[0].get("answer_text") or ""))
                    for line in missed_requirements(g, cs):
                        print(f"             missed: {line}")
                if not n_fresh:
                    records.append({"gold": g, "runs": cached_runs})
                    continue

            runs = list(cached_runs)
            for r in range(len(cached_runs), len(cached_runs) + n_fresh):
                try:
                    answer, trace = _call_agent(g.question, col,
                                                quiet=not args.show_agent,
                                                no_rag=args.no_rag)
                except Exception as e:
                    # One question/model failing (no tool support, invalid model
                    # id, rate limit that outlasted the backoff) shouldn't abort
                    # the batch — record nothing for this run and move on.
                    # 300 chars so OpenRouter's nested upstream error is visible.
                    print(f"      run {r+1}: ⚠️  agent error: {type(e).__name__}: {str(e)[:300]}")
                    continue
                pool = retrieval_pool(trace)
                ans = cited_verses(answer)

                if g.is_refusal:
                    refused = float(is_refusal(answer))
                    runs.append({"n_calls": len(trace), "answer_text": answer,
                                 "refusal_pass": refused})
                    tag = "✅" if refused else "❌"
                    print(f"      run {r+1}: {tag} refused={refused:.0f} "
                          f"(cited {len(ans)} verses, {len(trace)} calls)")
                    continue

                if args.no_rag:
                    fa = norag_faithfulness(answer, version)
                    runs.append({
                        "n_calls":     len(trace),                   # 0 (no retrieval)
                        "answer_text": answer,                       # raw, for re-scoring
                        "answer":      score_candidate(g, ans),
                        "norag_faith": fa,
                    })
                else:
                    runs.append({
                        "n_calls":     len(trace),
                        "answer_text": answer,                          # raw, for re-scoring
                        "pool_refs":   [format_ref(t) for t in pool],   # ranked, for re-scoring
                        "pool":        score_candidate(g, pool),
                        "answer":      score_candidate(g, ans),
                        "faithful":    faithfulness(answer, pool),
                        "pool_top":    [format_ref(t) for t in pool[:3]],
                    })
                answer_hit = runs[-1]["answer"]["ceiling"]
                tag = "✅" if answer_hit else "❌"
                # Show what the score judges — the verses the agent SHOWED —
                # not the cosine-ranked pool top (often irrelevant near-misses).
                first = ", ".join(format_ref(t) for t in ans[:3]) or "—"
                more = f" +{len(ans) - 3} more" if len(ans) > 3 else ""
                extra = f", {len(trace)} calls, shown: {first}{more}"
                if args.no_rag and runs[-1].get("norag_faith"):
                    fa = runs[-1]["norag_faith"]
                    extra = (f", shown: {first}{more}, faithful "
                             f"{fa['faithful']*100:.0f}% of {fa['n_cited']}")
                print(f"      run {r+1}: {tag} answer_hit={answer_hit:.0f} ({extra.lstrip(', ')})")
                if not answer_hit:
                    for line in missed_requirements(g, set(ans)):
                        print(f"             missed: {line}")

            records.append({"gold": g, "runs": runs})
            # Persist after every question — an interrupt or crash loses at most
            # the question in flight, and the next invocation resumes from here.
            _save(records, version, args, gold_csv, announce=False)

            if len(runs) == len(cached_runs):       # every fresh attempt errored
                consec_failed += 1
                if consec_failed >= 3:
                    print("\n🛑  3 questions in a row failed every attempt — stopping this "
                          "model's eval early. Partial results are saved; re-run the same "
                          "command to resume.")
                    break
            else:
                consec_failed = 0
    except KeyboardInterrupt:
        print("\n🛑  Interrupted — completed questions are already saved; "
              "re-run the same command to resume.")

    _report(records, version, args.runs, time.time() - t0, args.detailed, args.no_rag)
    _save(records, version, args, gold_csv)


class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def _report(records, version, runs, elapsed, detailed=False, no_rag=False):
    line = "─" * 72
    scored = [r for r in records if not r["gold"].is_refusal]   # verse-gold questions
    refusals = [r for r in records if r["gold"].is_refusal]
    mode = "  [NO-RAG baseline]" if no_rag else ""
    print(f"\n{line}\n  RESULTS — {MODEL_NAME} on {version}{mode}, {len(records)} questions × {runs} run(s), {elapsed:.0f}s")
    print(f"  {len(scored)} verse-gold + {len(refusals)} refusal\n{line}")

    print("\n  Answer hit rate — do the verses the agent SHOWS contain the gold?")
    print("    (end-to-end; scored over every shown verse, no cutoff)")
    print(f"    overall: {_fmt(_agg(scored,'answer','ceiling'))}")

    if no_rag:
        # No retrieval to report; instead check the recited citations for fabrication.
        fa = lambda k: _mean([_mean([r["norag_faith"][k] for r in rec["runs"]
                                     if r.get("norag_faith")]) for rec in scored])
        faithful = fa("faithful")
        print("\n  Faithfulness — do the model's OWN quoted verses match the real text?")
        print("    (no retrieval; RAG copies text verbatim, so this is a no-RAG-only check)")
        print(f"    reference valid (verse exists):     {_fmt(fa('ref_valid'))}")
        print(f"    faithful (ref exists AND text ok):  {_fmt(faithful)}")
        if faithful is not None:
            print(f"    → hallucination rate:               {_fmt(1 - faithful)}")
    else:
        print("\n  Retrieval hit rate — is a required verse in the top-k of the search?")
        for k in KS:
            print(f"    top-{k:<4}{_fmt(_agg(scored,'pool',f'pass@{k}')):>8}")
        print(f"    found anywhere in pool (ceiling): {_fmt(_agg(scored,'pool','ceiling'))}")

    if refusals:
        print(f"\n  Refusal accuracy (correctly declines out-of-scope): "
              f"{_fmt(_refusal_acc(refusals))}  ({len(refusals)} q)")

    print("\n  By category (answer hit rate; refusal accuracy for the refusal row)")
    cats: dict[str, list] = {}
    for rec in records:
        cats.setdefault(rec["gold"].category, []).append(rec)
    for cat in sorted(cats, key=cat_sort_key):
        recs = cats[cat]
        val = _refusal_acc(recs) if cat == "refusal" else _agg(recs, "answer", "ceiling")
        print(f"    {cat:<14}{_fmt(val):>8}  (n={len(recs)})")

    if detailed and not no_rag:
        print("\n  Diagnostics (retrieval pool)")
        print(f"    Ceiling (gold found anywhere):  {_fmt(_agg(scored,'pool','ceiling'))}")
        print(f"    MRR (first gold verse):         {_fmt(_agg(scored,'pool','mrr'), pct=False)}")
        for k in KS:
            print(f"    Graded recall@{k} (answer-all):  {_fmt(_agg(scored,'pool',f'recall_all@{k}'))}")
        faith = _mean([_mean([r["faithful"] for r in rec["runs"]]) for rec in scored])
        calls = _mean([_mean([r["n_calls"] for r in rec["runs"]]) for rec in records])
        print(f"    Faithfulness (cited ⊆ found):   {_fmt(faith)}")
        print(f"    Avg retrieval calls / question: {_fmt(calls, pct=False)}")

    print("\n  Per-question (answer hit rate, or refusal; mean over runs)")
    for rec in records:
        g = rec["gold"]
        if g.is_refusal:
            v = _mean([r["refusal_pass"] for r in rec["runs"]])
        else:
            v = _mean([r["answer"]["ceiling"] for r in rec["runs"]])
        flag = " ⚑" if g.comment else ""
        print(f"    {_fmt(v):>7}  [{g.category[:8]:<8}] {g.question}{flag}")

    _print_markdown(records)
    print(line)


def _print_markdown(records) -> None:
    """A copy-paste Markdown row for this model — the resume/README artifact.
    Use `python -m evaluation.leaderboard` to combine models into one table."""
    questions = [{"category": r["gold"].category, "runs": r["runs"]} for r in records]
    s = summarize(questions)
    cats = sorted(s["by_category"], key=cat_sort_key)

    print("\n  Markdown (paste into README / resume):\n")
    print("    | Model | n | Answer hit | Retrieval@3 | Retrieval@10 | Refusal |")
    print("    |---|---|---|---|---|---|")
    print(f"    | {MODEL_NAME} | {s['n']} | {md_pct(s['answer_hit'])} | "
          f"{md_pct(s['retr@3'])} | {md_pct(s['retr@10'])} | {md_pct(s['refusal'])} |")

    print(f"\n    | Model | {' | '.join(cats)} |")
    print(f"    |---|{'|'.join(['---'] * len(cats))}|")
    print(f"    | {MODEL_NAME} | "
          f"{' | '.join(md_pct(s['by_category'][c]) for c in cats)} |")


def _save(records, version, args, csv_gold, announce=True):
    """
    Merge this run into the stable per-(version, model) results file.

    The filename carries no timestamp, so re-runs update in place. Entries for
    questions no longer in the CSV are dropped; questions in the CSV but not in
    this run's selection keep their stored answers (re-scored from raw), so a
    --limit run never erases previous full-run results. An errored fresh run
    (no successful runs) never overwrites previously stored answers.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _result_path(version, args.csv, args.no_rag)
    stored = load_cache(path)
    fresh = {rec["gold"].question: rec["runs"] for rec in records}

    questions = []
    for g in csv_gold:                      # CSV order → stable, diffable files
        if g.question not in fresh and g.question not in stored:
            continue
        runs = fresh.get(g.question) or []
        if not runs and stored.get(g.question, {}).get("runs"):
            runs = [rescore_run(g, r, args.no_rag, version) for r in stored[g.question]["runs"]]
        questions.append({
            "question": g.question,
            "category": g.category,
            "comment": g.comment,
            "answer_all": g.raw_all,
            "answer_any": g.raw_any,
            "answer_any_two": g.raw_any_two,
            "answer_any_plus": g.raw_any_plus,
            "runs": runs,
        })

    payload = {
        "model": MODEL_NAME, "version": version,
        "mode": "no_rag" if args.no_rag else "rag",
        "runs": max((len(q["runs"]) for q in questions), default=args.runs),
        "csv": args.csv,
        "reasoning_effort": REASONING_EFFORT,
        "timestamp": datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
        "questions": questions,
    }
    path.write_text(json.dumps(payload, indent=2))
    if announce:
        print(f"\n💾  Results → {path}  ({len(questions)}/{len(csv_gold)} CSV questions stored)")


if __name__ == "__main__":
    main()
