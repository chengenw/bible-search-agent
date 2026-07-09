# Bible Semantic Search Agent

A tool-using LLM agent that answers Bible questions **only** from a local vector
database — not from the model's own memory — plus an **evaluation harness**
that benchmarks multiple models and asks the question many RAG demos skip:
*does retrieval actually beat what the model already knows?*

> **What this project demonstrates:** an agentic tool-use loop, retrieval-augmented
> generation with strict grounding, a from-scratch evaluation harness (multi-metric,
> cached, resumable, interrupt-safe), multi-model benchmarking, and a
> hallucination/faithfulness metric — built and stress-tested against several
> hosted models.

**Finding:** on a *memorized* public text like the Bible, a grounded RAG agent
does **not** out-score a strong model's own recall — retrieval is capped by
search quality (~66% top-10), below the model's memory (~91%). What retrieval
adds here is **grounding**: every citation is verbatim from the source, whereas
the same models answering from memory drift between translations and occasionally
misquote. The harness exists to measure that trade-off — where retrieval helps
and where it doesn't — rather than assuming it always does.

```
❓  who is Melchizedek?

🔍  [turn 1: 1/4] Searching: 'Melchizedek king of Salem'
🔍  [turn 1: 2/4] Searching: 'priest order of Melchizedek'

**Relevant Verses:**
1. [Genesis 14:18] — And Melchizedek king of Salem brought forth bread and wine…
2. [Psalms 110:4] — The LORD hath sworn… Thou [art] a priest for ever after the order of Melchizedek.
3. [Hebrews 7:1] — For this Melchisedec, king of Salem, priest of the most high God…

**Summary:**
Melchizedek was king of Salem and a priest of the most high God who blessed Abraham.
Scripture speaks of an eternal "order of Melchizedek," and identifies Jesus as a high
priest forever after it — distinct from the Levitical priesthood.
```

---

## What I learned

- **Building the eval taught more than the agent did.** A rigorous harness with
  gold labels is what let me *catch* that RAG doesn't beat memory here — and
  explain why (retrieval ceiling < model recall).
- **Measuring hallucination is subtle.** My first faithfulness check reported
  **42%** — but inspection showed almost all of it was the model reciting real
  verses in *NIV/ESV* wording, scored against *KJV*. Checking each quote against
  **every cached translation** (KJV/ESV/NASB/NIV/NLT) brought it down to **~2%
  for capable models** — and that residual is paraphrase or a verse *described*
  rather than quoted; reference validity stays 100%, so they never invent a verse
  (a small model like gpt-oss-20b scores worse only because it paraphrases
  loosely, not because it fabricates). A metric that conflates "different
  translation" with "made up" is worse than no metric — a lesson that generalizes
  to any grounded-generation eval.
- **Model output is untrusted input.** Benchmarking many models surfaced real
  misbehaviour — one emitted 24 *parallel* tool calls in a single turn and
  ballooned a request to 537k tokens. The agent loop now defends against it
  (see below).
- **Where RAG *would* win** (and the natural next step): a private/unseen corpus,
  a need for exact text in a chosen translation, or a small model that can't
  recite scripture — all cases where the "answer from memory" baseline collapses.

---

## Architecture

```
User question
     │
     ▼
┌─────────────────────────────────────┐
│  Agent loop  (litellm + tool use)   │
│  1. Force first retrieval call      │◄── tool_choice="function" on turn 1
│  2. Model issues 1–4 search calls   │◄── short, verse-like queries
│  3. Cap at MAX_RETRIEVAL_CALLS=4    │◄── then force a grounded answer
└─────────────┬───────────────────────┘
              │ retrieve_verses(query, testament?, book?)
              ▼
┌─────────────────────────────────────┐
│  ChromaDB (persistent, cosine)      │
│  BAAI/bge-base-en-v1.5, ~31k verses │◄── metadata-filtered by testament/book
└─────────────┬───────────────────────┘
              │ cache miss → download
              ▼
┌─────────────────────────────────────┐
│  Bolls.life REST API (13 versions)  │◄── pickled locally after first fetch
└─────────────────────────────────────┘
```

**Stack:** `bge-base-en-v1.5` embeddings (better archaic-English recall than
MiniLM) · ChromaDB (local, no server) · litellm (provider-agnostic — swap models
via `.env`) · CPU-only PyTorch (no GPU required).

---

## Engineering highlights

- **Grounding is structural, not just prompted.** `tool_choice="function"` on turn 1
  makes retrieval mandatory — the model *cannot* answer before searching — and the
  system prompt forbids citing any reference the tool didn't return.
- **Retrieval budget enforced per call, not per turn.** The agent gets 4 searches,
  then a final answer is forced. Crucially the cap counts *individual* tool calls,
  because one model response can carry many parallel calls — a benchmarked run
  recorded 24 in a single turn before this cap existed — each appending verse
  JSON to the history until it overflowed the context window.
- **Model output treated as untrusted:** history is clipped so it can't snowball;
  malformed-JSON and unknown-tool calls get corrective replies instead of crashing;
  `max_tokens` is always set; transient 429s/timeouts retry with backoff (litellm
  doesn't by default).
- **Query reformulation for recall.** The prompt rewrites natural questions
  ("Can Jews pray outside Jerusalem?") into short verse-like fragments
  ("pray toward Jerusalem exile") that embed closer to the target text.

---

## Evaluation

`evaluation/` scores the agent against a labelled question set (`verses.csv`)
using the same model/key from `.env`. The metrics separate the retriever's job
from the LLM's:

- **Answer hit rate** — do the verses the agent *shows the user* contain every
  required verse? (end-to-end, scored over all shown verses).
- **Retrieval@3 / @10** — how highly the *cosine search* ranks the gold in its
  pool (the embedding's ranking, before the LLM chooses). Answer-hit usually
  *exceeds* Retrieval@10 — the LLM surfaces relevant verses the raw ranking
  buried; the shortfall from the pool *ceiling* (gold found anywhere) is the
  LLM's selection loss.
- **No-RAG** — same questions with no retrieval (answer from memory); the gap to
  answer-hit is the measured RAG benefit.
- **No-RAG faithful** — of the model's own recited citations, what fraction's text
  actually matches the real verse (checked across every cached translation). The
  shortfall is its hallucination rate. (RAG is verbatim, so ~100% by construction.)

```bash
python -m evaluation.run_eval                      # full run, 3 runs/question, averaged
python -m evaluation.run_eval --limit 12 --runs 1  # quick, spans all categories
python -m evaluation.run_eval --no-rag             # memory-only baseline
./evaluation/eval_models.sh --runs 1               # whole model roster → leaderboard
python -m evaluation.leaderboard                   # combined Markdown leaderboard
```

Example leaderboard:

| Model | Ver | Runs | n | Answer hit | No-RAG | No-RAG faithful | Retrieval@3 | Retrieval@10 | Refusal |
|---|---|---|---|---|---|---|---|---|---|
| minimax-m3 | KJV | 3 | 50 | 91% | 96% | 97% | 52% | 66% | 100% |
| mimo-v2.5-pro | KJV | 3 | 50 | 91% | 95% | 98% | 41% | 65% | 100% |
| deepseek-v4-flash | KJV | 3 | 50 | 90% | 92% | 97% | 48% | 65% | 100% |
| qwen3-235b-a22b-2507 | KJV | 3 | 50 | 80% | 91% | 98% | 37% | 53% | 100% |
| gpt-oss-20b | KJV | 3 | 50 | 51% | 82% | 21% | 26% | 41% | 100% |

**Results are cached and runs resume.** Answers are stored per
(translation, model) and re-scored from their raw text on *every* run — so
editing a gold label or the parser and simply re-running updates the scores with
*zero* new LLM calls (`--reeval` is only for re-fetching fresh answers, e.g.
after changing the agent's prompt or model). A crash or Ctrl-C loses at most the
question in flight. Questions use a small CSV schema (`Answer all` / `any` /
`any two` / `any plus` gold columns for all-of / any-of / two-of /
both-sides-of-a-debate questions). Full flag reference and mechanics live in the
module docstrings.

---

## Setup & usage

```bash
uv sync && source .venv/bin/activate
```

Create `.env` — litellm picks the provider from the model-string prefix, so
switching providers is two lines:

```
OPENROUTER_API_KEY=sk-or-...
MODEL=openrouter/deepseek/deepseek-v4-flash   # or bedrock/…, gpt-4o-mini, ollama/…, etc.
```

```bash
python main.py                    # KJV (default); first run downloads + indexes (~5 min)
python main.py --version ESV      # any of KJV·ASV·WEB·YLT·NASB·NIV·ESV·NKJV·NLT·MSG·RSV·NET·AMP
```

The model must support tool calling (the agent forces a tool call on turn 1).
Optional: `REASONING_EFFORT=low|medium|high`, `MAX_COMPLETION_TOKENS`.

---

## Limitations

- **Retrieval depends on the agent's own query decomposition** — a missed facet is
  simply never retrieved, and the agent can't know what it missed.
- **The summary is LLM-generated** — the retrieved verses are ground truth; verify
  the summary against them.
- **Pure semantic search, single verses, no keyword fallback** — by design;
  keyword/exact-phrase and passage context are well served by existing tools.
- **Requires an LLM API key** (a local Ollama model works via litellm but with
  lower quality). CLI only.
