"""
main.py — CLI entry point for the Bible Semantic Search Agent.

Usage:
    python main.py                        # KJV (default)
    python main.py --version ESV          # Any supported translation
    python main.py --version KJV --build  # Force rebuild the vector index

Supported versions: KJV ASV WEB YLT NASB NIV ESV NKJV NLT MSG RSV NET AMP
"""

import argparse

from src.config import MODEL, VERSIONS
from src.vector_store import get_collection
from src.agent import run_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Bible Semantic Search Agent")
    parser.add_argument(
        "--version", default="KJV",
        choices=list(VERSIONS),
        help="Bible translation to search (default: KJV)",
    )
    parser.add_argument(
        "--build", action="store_true",
        help="Force rebuild the vector index from scratch",
    )
    args    = parser.parse_args()
    version = args.version.upper()

    print(f"\n📚  Version: {version}  |  Model: {MODEL}  |  Source: Bolls API\n")
    col = get_collection(version, force_rebuild=args.build)

    print("=" * 65)
    print(f"  Bible Semantic Search Agent  [{version}]")
    print("  Type your question, or 'quit' to exit.")
    print("=" * 65)

    while True:
        try:
            question = input("\n❓  Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        print()
        answer = run_agent(question, col)
        print("\n" + answer)
        print("\n" + "─" * 65)


if __name__ == "__main__":
    main()
