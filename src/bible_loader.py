"""
bible_loader.py — Download verses from Bolls API and cache locally as pickle.
"""

import pickle
import re
import time
from pathlib import Path

import requests

from src.config import (
    BOLLS_BOOKS_URL,
    BOLLS_VERSES_URL,
    CACHE_BASE,
    VERSIONS,
    get_testament,
)


def _download_from_bolls(version_key: str) -> list[dict]:
    """
    Fetch all verses for a translation from the Bolls.life free REST API.
    Politely rate-limited at 0.1 s per chapter request (~5 min for a full Bible).
    """
    slug = VERSIONS[version_key]
    print(f"📖  Downloading {version_key} from Bolls API (one-time, ~5 min)...")

    resp = requests.get(BOLLS_BOOKS_URL.format(translation=slug), timeout=30)
    resp.raise_for_status()
    books = resp.json()

    verses      = []
    total_books = len(books)

    for bi, book in enumerate(books):
        book_id   = book["bookid"]
        book_name = book["name"]
        num_chaps = book["chapters"]

        if book_id > 66:      # skip Apocrypha; Bolls numbers the 66 canonical books 1–66
            continue

        for chapter in range(1, num_chaps + 1):
            try:
                r = requests.get(
                    BOLLS_VERSES_URL.format(
                        translation=slug, book=book_id, chapter=chapter
                    ),
                    timeout=30,
                )
                r.raise_for_status()
                for v in r.json():
                    # Some Bolls editions tag Strong's numbers as <S>1234</S>;
                    # drop those (number and all), then strip any other markup.
                    text      = re.sub(r"<S>\d+</S>", "", v.get("text", ""))
                    text      = re.sub(r"<[^>]+>", "", text)
                    text      = re.sub(r"\s+", " ", text).strip()
                    verse_num = v.get("verse", 0)
                    if text:
                        verses.append({
                            # zero-padded book/chapter/verse → a stable, sortable id
                            "id":        f"{book_id:02d}{chapter:03d}{verse_num:03d}",
                            "text":      text,
                            "book":      book_name,
                            "testament": get_testament(book_name),
                            "chapter":   chapter,
                            "verse":     verse_num,
                            "reference": f"{book_name} {chapter}:{verse_num}",
                        })
                time.sleep(0.1)
            except Exception as e:
                print(f"    Warning: {book_name} {chapter} — {e}")

        pct = int((bi + 1) / total_books * 100)
        print(f"    {pct}%  {book_name:<30}", end="\r")

    print(f"\n    Downloaded {len(verses):,} verses.")
    return verses


def load_all_verses(version_key: str) -> list[dict]:
    """
    Return all verses for the given translation.
    Downloads from Bolls API on first call; subsequent calls load from pickle cache.
    """
    Path(CACHE_BASE).mkdir(parents=True, exist_ok=True)
    cache_file = Path(CACHE_BASE) / f"{version_key}.pkl"

    if cache_file.exists():
        print(f"📖  Loading {version_key} from cache...")
        with open(cache_file, "rb") as f:
            verses = pickle.load(f)
        if verses:
            print(f"    {len(verses):,} verses ready.")
            return verses
        print("⚠️   Cache empty — re-downloading...")
        cache_file.unlink()

    if version_key not in VERSIONS:
        raise ValueError(
            f"Unknown version '{version_key}'. "
            f"Available: {', '.join(VERSIONS)}"
        )

    verses = _download_from_bolls(version_key)

    with open(cache_file, "wb") as f:
        pickle.dump(verses, f)

    return verses
