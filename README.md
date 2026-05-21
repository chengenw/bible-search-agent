# Bible Semantic Search Agent

A RAG-based agent that answers questions about the Bible by searching a local vector database — designed to return only what the database supports, not from the model's own knowledge.

For example:

```
❓  Your question: who is Melchizedek?

🔍  [turn 1: 1/4] Searching: 'Melchizedek'
🔍  [turn 1: 2/4] Searching: 'Melchizedek king of Salem'
🔍  [turn 1: 3/4] Searching: 'Melchizedek priest'
🔍  [turn 1: 4/4] Searching: 'priest order of Melchizedek'

**Relevant Verses:**
1. [Genesis 14:18] — And Melchizedek king of Salem brought forth bread and wine: and he [was] the priest of the most high God.
2. [Psalms 110:4] — The LORD hath sworn, and will not repent, Thou [art] a priest for ever after the order of Melchizedek.
3. [Hebrews 5:6] — As he saith also in another [place], Thou [art] a priest for ever after the order of Melchisedec.
4. [Hebrews 6:20] — Whither the forerunner is for us entered, [even] Jesus, made an high priest for ever after the order of Melchisedec.
5. [Hebrews 7:1] — For this Melchisedec, king of Salem, priest of the most high God, who met Abraham returning from the slaughter of the kings, and blessed him;
6. [Hebrews 7:11] — If therefore perfection were by the Levitical priesthood, ( for under it the people received the law,) what further need [was there] that another priest should rise after the order of Melchisedec, and not be called after the order of Aaron?
7. [Hebrews 7:15] — And it is yet far more evident: for that after the similitude of Melchisedec there ariseth another priest,

**Summary:**
Based on the verses provided, Melchizedek was the king of Salem and a priest of the most high God. He met Abraham, blessed him, and brought out bread and wine. There is an "order of Melchizedek," which is a priesthood that is eternal. The scriptures mention another priest arising in the likeness of Melchizedek, and Jesus is identified as a high priest forever according to this order. This priesthood is presented as distinct from the Levitical priesthood.
```

---

## Motivation

Most Bible study tools are keyword search. Searching "faith" returns verses
containing the word "faith" — but misses verses about trusting God that don't
use that word. This agent searches by *meaning*.

Verses are returned first, with a brief LLM-generated summary below. The LLM
is grounded strictly to the local database — the same pattern used in
enterprise RAG systems where an LLM must answer only from a specific knowledge
base, not its general training.

---

## Architecture

```
User question
     │
     ▼
┌─────────────────────────────────────┐
│  Agent loop  (litellm + tool use)   │
│                                     │
│  1. Force first retrieval call      │◄── tool_choice="function" on turn 1
│  2. Model issues 1–4 search calls   │◄── short, verse-like queries
│  3. Cap at MAX_RETRIEVAL_CALLS=4    │◄── then force final answer
│  4. Produce grounded answer         │
└─────────────┬───────────────────────┘
              │ retrieve_verses(query, testament?, book?)
              ▼
┌─────────────────────────────────────┐
│  ChromaDB  (persistent, cosine)     │
│  BAAI/bge-base-en-v1.5 embeddings   │
│  ~31,000 verses, metadata filtered  │
└─────────────┬───────────────────────┘
              │ cache miss → download
              ▼
┌─────────────────────────────────────┐
│  Bolls.life REST API                │
│  13 translations, no API key        │
│  pickled locally after first fetch  │
└─────────────────────────────────────┘
```

---


## Key Design Decisions

### 1. Hallucination prevention

Two defenses work together:

- `tool_choice="function"` on turn 1 makes retrieval *structurally mandatory*;
  the model cannot respond before searching
- The system prompt instructs the model not to cite any reference not returned
  by the tool, and not to quote or paraphrase from memory

### 2. Query design: short, verse-like phrases improve recall over natural questions

A natural question like *"Can Jews pray outside Jerusalem?"* embeds very
differently from the Bible passage it's looking for. The system prompt
instructs the agent to rephrase queries as short, verse-like fragments —
*"pray toward Jerusalem exile"* — which improves recall on both archaic
and modern translations.

### 3. Metadata pre-filtering before vector search

ChromaDB `where` filters are applied *before* the vector search, not as
post-processing. This means:

- Testament-scoped queries only search the ~23,000 OT or ~8,000 NT verses
- Book-scoped queries only search ~1,000 verses in that book
- `top_k=25` — results stay within the intended scope

### 4. Retrieval budget with forced termination

The agent allows up to `MAX_RETRIEVAL_CALLS = 4` tool calls, then forces a
final answer via `tool_choice="none"`. This prevents infinite retrieval loops
without truncating the agent's ability to issue multiple targeted searches —
multi-faceted questions often need 2–3 differently-phrased queries to surface
the best verses.

### 5. Embedding model: bge-base-en-v1.5 over MiniLM

BGE-base handles formal and archaic register better than MiniLM for this
domain, at the cost of slightly longer CPU build time.

---


## Stack

| Component | Choice | Why |
|---|---|---|
| Embeddings | `BAAI/bge-base-en-v1.5` | Better archaic/formal English recall than MiniLM |
| Vector DB | ChromaDB (persistent) | Lightweight, local, no server needed |
| Agent / LLM | litellm (model-agnostic) | Swap OpenAI / Anthropic / local via `.env` |
| Bible data | Bolls.life REST API | Free, no key, 13 translations |
| Runtime | CPU-only PyTorch | No GPU required |

---


## Project structure

```
bible-search-agent/
├── main.py              # CLI entry point
├── src/
│   ├── config.py        # Constants, version map, book sets, name aliases
│   ├── bible_loader.py  # Bolls API download + pickle cache
│   ├── vector_store.py  # ChromaDB collection build/load + retrieve_verses()
│   └── agent.py         # Tool schema, system prompt, run_agent()
└── data/
    ├── bible_db/        # ChromaDB vector indexes (one per translation)
    └── bible_cache/     # Pickle cache of raw verse downloads
```

---


## Setup

```bash
git clone <this-repository>
cd <this-repository>
uv sync
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install sentence-transformers
source .venv/bin/activate
```

Create a `.env` file:

```
OPENAI_API_KEY=sk-...
MODEL=gpt-4o-mini        # or gpt-4o, claude-3-5-haiku-20241022, etc.
```

---

## Usage

```bash
# KJV (default) — downloads and indexes on first run (~5 min), instant after
python main.py

# Different translation
python main.py --version ESV

# Force rebuild the vector index
python main.py --version KJV --build
```

**Supported translations:** KJV · ASV · WEB · YLT · NASB · NIV · ESV · NKJV · NLT · MSG · RSV · NET · AMP

Each search runs within one translation. Verses are fetched from [Bolls.life](https://bolls.life/api/) (free REST API, no key required), cached locally as pickle files, and indexed once per translation via ChromaDB.

---

## Example questions

```
What does Proverbs say about pride?
What are spiritual gifts?
Must Jews pray to God through a priest in the old testament?
Are Christian free to sin based on Romans 3:28?
Is it possible for Christians to lose their salvation?
```

---


## Limitations

**Retrieval coverage depends on how well the agent decomposes the question.** The agent issues 1–4 short search queries of its own choosing. If it picks poor angles or misses an important facet of the question, relevant verses simply won't be retrieved — and the agent has no way to know what it missed. For complex questions, try splitting into separate focused questions, or name the specific book or theme you want searched.

**The summary is LLM-generated and may not always be accurate.** The retrieved verses are the ground truth; treat the summary as a navigational aid and verify anything that matters against the Bible directly.

**Pure semantic search — no keyword fallback.** This is intentional. Keyword
and exact-phrase search is well-served by existing tools (concordances, Bible
apps). This project's purpose is to find thematically relevant verses even
when no keyword matches. Combining the two is out of scope.

**Results are individual verses, without surrounding context.** A verse can read differently in isolation than in its passage. This is true of any concordance-style tool — for any result that matters, read the surrounding chapter.

**Requires an LLM API key.** The agent depends on an external API (OpenAI,
Anthropic, etc.) for query reformulation and summarization. A local
alternative is possible — [Ollama](https://ollama.com) running a small model
(`llama3.2:3b`, `mistral`) works with litellm via `MODEL=ollama/llama3.2` and
no API key — but retrieval quality and summary accuracy will be lower.

**CLI only.** There is no web UI. Users need to be comfortable running a
terminal.
