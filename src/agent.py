"""Tool schema, system prompt, and the retrieval-augmented agent loop."""

import json

import litellm
import chromadb

from src.config import (
    MODEL,
    MAX_RETRIEVAL_CALLS,
    MAX_COMPLETION_TOKENS,
    REASONING_EFFORT,
)
from src.vector_store import retrieve_verses

litellm.drop_params = True            # skip params a given model doesn't support
litellm.suppress_debug_info = True    # silence litellm's "Provider List" banners

# Applied to every completion call. litellm doesn't retry or set a sane timeout
# by default, so one 429 or a hung provider would otherwise kill/freeze a run.
_LLM_KWARGS: dict = {
    "max_tokens":  MAX_COMPLETION_TOKENS,
    "num_retries": 2,
    "timeout":     120,
}
if REASONING_EFFORT:
    # litellm's OpenRouter mapping drops reasoning_effort, so pass it via extra_body.
    if MODEL.startswith("openrouter/"):
        _LLM_KWARGS["extra_body"] = {"reasoning": {"effort": REASONING_EFFORT}}
    else:
        _LLM_KWARGS["reasoning_effort"] = REASONING_EFFORT

# Dispatch by name so a hallucinated tool name returns an error to the model
# rather than raising.
TOOL_DISPATCH = {
    "retrieve_verses": retrieve_verses,
}

# Cap what a misbehaving model can stuff into the (re-sent-every-turn) history:
# huge reasoning dumps or runaway arguments would otherwise blow the context limit.
MAX_HISTORY_CONTENT = 4000   # chars of assistant prose
MAX_HISTORY_ARGS    = 2000   # chars of tool-call arguments


def _clip(text, limit: int) -> str:
    """Truncate history text past `limit`, marking how much was cut."""
    text = text or ""
    if len(text) > limit:
        return text[:limit] + f" …[truncated {len(text) - limit} chars]"
    return text


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_verses",
            "description": (
                "Search the Bible for verses relevant to a topic or question. "
                "Supports optional scoping to Old Testament, New Testament, or a specific book. "
                "Always call this before answering any Bible question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "SHORT, concrete, verse-like query — 3-6 words max. "
                            "Phrase it like Bible text, not like a question. "
                            "Good: 'pray toward Jerusalem exile'  "
                            "Good: 'Daniel prays Babylon'  "
                            "Bad:  'Can Jews pray outside Jerusalem?'"
                        ),
                    },
                    "testament": {
                        "type": "string",
                        "enum": ["Old", "New"],
                        "description": (
                            "Restrict search to one testament. "
                            "Use 'Old' when the question mentions Old Testament, Hebrew Bible, "
                            "Torah, Prophets, or refers to pre-Christian times. "
                            "Use 'New' for questions about Jesus, apostles, or New Testament. "
                            "Omit to search the whole Bible."
                        ),
                    },
                    "book": {
                        "type": "string",
                        "description": (
                            "Restrict search to a single Bible book by its full name, "
                            "e.g. 'Matthew', 'Daniel', 'Psalms'. "
                            "Use when the question explicitly mentions a specific book. "
                            "Omit otherwise."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    }
]

SYSTEM_PROMPT = """You are a Bible study assistant that answers ONLY from retrieved verses.

STRICT RULES — never break these:
- NEVER quote, paraphrase, or invent a verse from your own knowledge.
- NEVER write a verse reference (e.g. "Daniel 6:10") unless it appeared in retrieval results.
- If no useful verses were returned, say exactly:
  "No relevant verses were found for this query. Try rephrasing your question."
  Do NOT supplement with verses from memory.
- Your summary must draw ONLY on the verses actually listed above it.

Stay neutral — no denominational bias:
- Do NOT advocate the position of any particular church, denomination, or tradition.
- Many questions are debated. When the retrieved verses support more than one position,
  present the verses for EACH side and let them stand — do not resolve the debate
  for the reader.
- Only present a side if verses for it were actually retrieved; never supply the
  "other side" from your own knowledge.

The Bible has two divisions:
  - Old Testament (39 books): Genesis → Malachi — Jewish scripture, pre-Christian.
  - New Testament (27 books): Matthew → Revelation — Christian scripture.

Retrieval strategy:
1. Identify testament/book scope; pass the correct filter to the tool.
2. Use SHORT, concrete queries (1-6 words) phrased like Bible text, not questions.
   Good: "pray toward Jerusalem exile"   Bad: "Can Jews pray outside Jerusalem?"
3. Issue 2-4 calls with DIFFERENT short queries to cover multiple angles.
   Stop once you have ≥3 clearly relevant results.
4. If the question is doctrinally contested, search EACH side separately so both
   are represented (e.g. "none shall pluck from my hand" AND "fall away crucify
   afresh" for eternal security).

Format:

**Relevant Verses:**
1. [Reference] — [verse text exactly as returned by the tool]
...

**Summary:**
[Synthesis based ONLY on the verses listed above — no outside knowledge.
 If the verses point to different positions, summarize each fairly without taking a side.]
"""

def run_agent(question: str, col: chromadb.Collection, return_trace: bool = False):
    """
    Run the retrieval-augmented agent loop.

    Until the first retrieval: tool_choice is forced, so the model cannot
    answer without searching. Then the model decides when to stop retrieving;
    after MAX_RETRIEVAL_CALLS, tool_choice="none" forces a final text answer.

    Returns the final answer string. When ``return_trace=True``, returns
    ``(answer, trace)`` where ``trace`` is a list of one dict per retrieval call —
    ``{"query", "testament", "book", "results"}`` — used by the evaluation harness.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]

    retrieval_calls = 0
    trace: list[dict] = []
    empty_responses = 0

    for iteration in range(10):  # safety cap — should never be reached
        if retrieval_calls == 0:
            # Force retrieval until at least one search has run — on turn 1,
            # and again if an empty errored turn was retried before any search.
            tool_choice = {"type": "function", "function": {"name": "retrieve_verses"}}
        elif retrieval_calls >= MAX_RETRIEVAL_CALLS:
            tool_choice = "none"
        else:
            tool_choice = "auto"

        response = litellm.completion(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice=tool_choice,
            **_LLM_KWARGS,
        )
        message = response.choices[0].message

        # Final text answer — done
        if message.content and not message.tool_calls:
            return (message.content, trace) if return_trace else message.content

        # Tool call(s)
        if message.tool_calls:
            empty_responses = 0          # a usable turn resets the empty streak
            # History is re-sent every turn, so clip prose/arguments before storing.
            messages.append({
                "role":       "assistant",
                "content":    _clip(message.content, MAX_HISTORY_CONTENT),
                "tool_calls": [
                    {
                        "id":       tc.id,
                        "type":     "function",
                        "function": {
                            "name":      tc.function.name,
                            "arguments": _clip(tc.function.arguments, MAX_HISTORY_ARGS),
                        },
                    }
                    for tc in message.tool_calls
                ],
            })

            for tc in message.tool_calls:
                if retrieval_calls >= MAX_RETRIEVAL_CALLS:
                    # Budget guard — one response may carry many parallel calls,
                    # and every id still needs a tool reply or the next request
                    # is rejected. Refuse the excess instead of executing it.
                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      ("Error: retrieval budget exhausted. "
                                         "Answer now using the verses already retrieved."),
                    })
                    continue

                tool_fn = TOOL_DISPATCH.get(tc.function.name)
                if tool_fn is None:
                    # LLM hallucinated a function name — return a clear error
                    # so the model can recover, rather than crashing the loop.
                    print(f"⚠️   Unknown tool requested by model: '{tc.function.name}'")
                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      (
                            f"Error: unknown tool '{tc.function.name}'. "
                            f"Available tools: {list(TOOL_DISPATCH)}"
                        ),
                    })
                    continue

                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    # Malformed arguments (truncated/garbage JSON) — tell the
                    # model instead of crashing the whole run.
                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      "Error: malformed JSON in tool arguments. "
                                        "Retry with valid JSON.",
                    })
                    continue
                query     = args.get("query", question)
                testament = args.get("testament")
                book      = args.get("book")

                scope = f" [book: {book}]" if book else (f" [testament: {testament}]" if testament else "")
                retrieval_calls += 1
                print(f"🔍  [turn {iteration + 1}: {retrieval_calls}/{MAX_RETRIEVAL_CALLS}] Searching: '{query}'{scope}")

                verses = tool_fn(col, query, testament=testament, book=book)
                trace.append({
                    "query":     query,
                    "testament": testament,
                    "book":      book,
                    "results":   verses,
                })
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      json.dumps(verses, indent=2),
                })
        else:
            # Neither content nor tool calls: some providers return an empty
            # errored completion (finish_reason "error"). Retry the turn — but
            # two empties IN A ROW means the provider is down; fall back then.
            empty_responses += 1
            if empty_responses >= 2:
                break

    fallback = "The agent could not produce an answer within the allowed steps."
    return (fallback, trace) if return_trace else fallback
