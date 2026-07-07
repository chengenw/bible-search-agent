"""Constants: model config, storage paths, the Bolls API, and Bible book sets."""

import os
from dotenv import load_dotenv

load_dotenv()

# rstrip(",") guards against a comma-separated model list being pasted into MODEL.
MODEL = os.getenv("MODEL", "gemini-2.5-pro").strip().rstrip(",")
TOP_K = 25   # retrieve a wide pool per query; the LLM shows the user ≤10
MAX_RETRIEVAL_CALLS = 4

# Reasoning depth for models that support it ("low"/"medium"/"high"); unset = default.
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "").strip().lower() or None

# Cap output length. Some providers otherwise default max_tokens to the model's
# full output window, which can overflow the context limit and 400.
MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS") or 8192)

DB_BASE = "./data/bible_db"
CACHE_BASE = "./data/bible_cache"

BOLLS_BOOKS_URL = "https://bolls.life/get-books/{translation}/"
BOLLS_VERSES_URL = "https://bolls.life/get-text/{translation}/{book}/{chapter}/"

VERSIONS = {
    "KJV":  "KJV",
    "ASV":  "ASV",
    "WEB":  "WEB",
    "YLT":  "YLT",
    "NASB": "NASB",
    "NIV":  "NIV",
    "ESV":  "ESV",
    "NKJV": "NKJV",
    "NLT":  "NLT",
    "MSG":  "MSG",
    "RSV":  "RSV",
    "NET":  "NET",
    "AMP":  "AMP",
}

OLD_TESTAMENT_BOOKS = {
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
    "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel",
    "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles",
    "Ezra", "Nehemiah", "Esther", "Job", "Psalms", "Proverbs",
    "Ecclesiastes", "Song of Solomon", "Isaiah", "Jeremiah",
    "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel", "Amos",
    "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah",
    "Haggai", "Zechariah", "Malachi",
}

NEW_TESTAMENT_BOOKS = {
    "Matthew", "Mark", "Luke", "John", "Acts",
    "Romans", "1 Corinthians", "2 Corinthians", "Galatians",
    "Ephesians", "Philippians", "Colossians",
    "1 Thessalonians", "2 Thessalonians",
    "1 Timothy", "2 Timothy", "Titus", "Philemon",
    "Hebrews", "James", "1 Peter", "2 Peter",
    "1 John", "2 John", "3 John", "Jude", "Revelation",
}

BOOK_NAME_ALIASES = {
    "song of songs":    "Song of Solomon",
    "canticles":        "Song of Solomon",
    "1 samuel":         "1 Samuel",
    "2 samuel":         "2 Samuel",
    "1 kings":          "1 Kings",
    "2 kings":          "2 Kings",
    "1 chronicles":     "1 Chronicles",
    "2 chronicles":     "2 Chronicles",
    "1 corinthians":    "1 Corinthians",
    "2 corinthians":    "2 Corinthians",
    "1 thessalonians":  "1 Thessalonians",
    "2 thessalonians":  "2 Thessalonians",
    "1 timothy":        "1 Timothy",
    "2 timothy":        "2 Timothy",
    "1 peter":          "1 Peter",
    "2 peter":          "2 Peter",
    "1 john":           "1 John",
    "2 john":           "2 John",
    "3 john":           "3 John",
}


def get_testament(book_name: str) -> str:
    """Return 'Old' or 'New' for a given book name."""
    if book_name in OLD_TESTAMENT_BOOKS:
        return "Old"
    if book_name in NEW_TESTAMENT_BOOKS:
        return "New"
    canonical = BOOK_NAME_ALIASES.get(book_name.lower().strip())
    if canonical:
        return get_testament(canonical)
    return "Unknown"
