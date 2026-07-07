"""
vector_store.py — Build and load the ChromaDB vector index.

Embedding model: BAAI/bge-base-en-v1.5 (CPU-only)
  - Chosen over all-MiniLM-L6-v2 for better recall on archaic/formal English (KJV).

Install:
    uv pip install torch --index-url https://download.pytorch.org/whl/cpu
    uv pip install sentence-transformers chromadb
"""

from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from src.bible_loader import load_all_verses
from src.config import DB_BASE, BOOK_NAME_ALIASES, TOP_K


def _make_embedding_function() -> SentenceTransformerEmbeddingFunction:
    return SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-base-en-v1.5",
        device="cpu",
    )


def get_collection(version_key: str, force_rebuild: bool = False) -> chromadb.Collection:
    """
    Return a persistent ChromaDB collection for the given Bible version.
    Builds the index on first call (a few minutes on CPU); instant on subsequent calls.
    Pass force_rebuild=True to wipe and re-index.
    """
    ef       = _make_embedding_function()
    db_path  = str(Path(DB_BASE) / version_key)
    col_name = f"bible_{version_key.lower()}"

    Path(db_path).mkdir(parents=True, exist_ok=True)
    client   = chromadb.PersistentClient(path=db_path)
    existing = [c.name for c in client.list_collections()]

    if col_name in existing and not force_rebuild:
        col = client.get_collection(name=col_name, embedding_function=ef)
        if col.count() > 0:
            print(f"✅  Loaded existing {version_key} index: {col.count():,} verses.\n")
            return col
        print(f"⚠️   Collection '{col_name}' exists but is empty — rebuilding...")

    if col_name in existing:
        client.delete_collection(col_name)

    col = client.create_collection(
        name=col_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    verses = load_all_verses(version_key)
    print(f"⚙️   Building vector index for {version_key} (CPU — a few minutes)...")

    batch_size = 500
    total      = len(verses)

    for i in range(0, total, batch_size):
        batch = verses[i : i + batch_size]
        col.add(
            ids       = [v["id"]   for v in batch],
            documents = [v["text"] for v in batch],
            metadatas = [{
                "reference": v["reference"],
                "book":      v["book"],
                "testament": v["testament"],
                "chapter":   v["chapter"],
                "verse":     v["verse"],
            } for v in batch],
        )
        pct = min(100, int((i + batch_size) / total * 100))
        print(f"    {pct}%  ({min(i + batch_size, total):,}/{total:,})", end="\r")

    print(f"\n✅  Index built: {col.count():,} verses.\n")
    return col


def retrieve_verses(
    col: chromadb.Collection,
    query: str,
    top_k: int = TOP_K,
    testament: str | None = None,
    book: str | None = None,
) -> list[dict]:
    """
    Semantic search with optional metadata pre-filtering.
    Filters are applied before the ANN search so top_k stays meaningful.

      testament="Old"  → only Old Testament verses
      testament="New"  → only New Testament verses
      book="Daniel"    → only that book
    """
    where = None
    if book:
        canonical = BOOK_NAME_ALIASES.get(book.lower(), book)
        where = {"book": canonical}
    elif testament:
        where = {"testament": testament}

    kwargs: dict = dict(query_texts=[query], n_results=top_k)
    if where:
        kwargs["where"] = where

    results = col.query(**kwargs)
    return [
        {
            "reference": meta["reference"],
            "testament": meta.get("testament", ""),
            "text":      doc,
            "score":     round(1 - dist, 4),   # cosine distance → similarity
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]
