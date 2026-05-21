"""
agent.py — Tool schema, system prompt, and the agent loop.
"""

import json

import litellm
import chromadb

from src.config import MODEL, MAX_RETRIEVAL_CALLS
from src.vector_store import retrieve_verses

# Explicit dispatch table — add entries here when new tools are introduced.
# Using a dict instead of globals() means an invented function name from the
# LLM produces a clear error message returned to the model, not a KeyError.
TOOL_DISPATCH = {
    "retrieve_verses": retrieve_verses,
}


# ─────────────────────────────────────────────
# Tool schema
# ─────────────────────────────────────────────

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

# ─────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Bible study assistant that answers ONLY from retrieved verses.

STRICT RULES — never break these:
- NEVER quote, paraphrase, or invent a verse from your own knowledge.
- NEVER write a verse reference (e.g. "Daniel 6:10") unless it appeared in retrieval results.
- If no useful verses were returned, say exactly:
  "No relevant verses were found for this query. Try rephrasing your question."
  Do NOT supplement with verses from memory.
- Your summary must draw ONLY on the verses actually listed above it.

The Bible has two divisions:
  - Old Testament (39 books): Genesis → Malachi — Jewish scripture, pre-Christian.
  - New Testament (27 books): Matthew → Revelation — Christian scripture.

Retrieval strategy:
1. Identify testament/book scope; pass the correct filter to the tool.
2. Use SHORT, concrete queries (1-6 words) phrased like Bible text, not questions.
   Good: "pray toward Jerusalem exile"   Bad: "Can Jews pray outside Jerusalem?"
3. Issue 2-4 calls with DIFFERENT short queries to cover multiple angles.
   Stop once you have ≥3 clearly relevant results.

Format:

**Relevant Verses:**
1. [Reference] — [verse text exactly as returned by the tool]
...

**Summary:**
[Synthesis based ONLY on the verses listed above — no outside knowledge]
"""

# ─────────────────────────────────────────────
# Agent loop
# ─────────────────────────────────────────────

def run_agent(question: str, col: chromadb.Collection) -> str:
    """
    Run the retrieval-augmented agent loop.

    Turn 1: tool_choice is forced to ensure at least one retrieval call.
    Turns 2–MAX_RETRIEVAL_CALLS: model decides whether to retrieve again or answer.
    After MAX_RETRIEVAL_CALLS: tool_choice="none" forces a final text answer.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]

    retrieval_calls = 0

    for iteration in range(10):  # safety cap — should never be reached
        if iteration == 0:
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
        )
        message = response.choices[0].message

        # Final text answer — done
        if message.content and not message.tool_calls:
            return message.content

        # Tool call(s)
        if message.tool_calls:
            messages.append({
                "role":       "assistant",
                "content":    message.content or "",
                "tool_calls": [
                    {
                        "id":       tc.id,
                        "type":     "function",
                        "function": {
                            "name":      tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            })

            for tc in message.tool_calls:
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

                args      = json.loads(tc.function.arguments)
                query     = args.get("query", question)
                testament = args.get("testament")
                book      = args.get("book")

                scope = f" [book: {book}]" if book else (f" [testament: {testament}]" if testament else "")
                retrieval_calls += 1
                print(f"🔍  [turn {iteration + 1}: {retrieval_calls}/{MAX_RETRIEVAL_CALLS}] Searching: '{query}'{scope}")

                verses = tool_fn(col, query, testament=testament, book=book)
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      json.dumps(verses, indent=2),
                })
        else:
            break

    return "The agent could not produce an answer within the allowed steps."
