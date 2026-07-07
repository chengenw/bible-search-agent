"""
refs.py — Bible reference normalization for evaluation.

Turns any reference form — gold-label abbreviations ("Pro 11:2", "2 cor 12:7-11"),
full names ("Genesis 14:18"), or the agent's own output ("Song of Songs 1:1") —
into canonical (book_id, chapter, verse) tuples so they can be compared exactly.

A *reference* may name a single verse or a continuous range. Per the eval spec, a
range counts as ONE gold entry that is satisfied if the agent retrieves ANY verse
inside it, so a range parses to the full list of its verse tuples and "satisfied"
means set-intersection is non-empty.
"""

import re

# Canonical 66-book Protestant order. Index + 1 == book_id.
# Names match the strings stored in the ChromaDB index (Bolls), e.g. "Song of Songs".
CANONICAL_BOOKS = [
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua",
    "Judges", "Ruth", "1 Samuel", "2 Samuel", "1 Kings", "2 Kings",
    "1 Chronicles", "2 Chronicles", "Ezra", "Nehemiah", "Esther", "Job",
    "Psalms", "Proverbs", "Ecclesiastes", "Song of Songs", "Isaiah", "Jeremiah",
    "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel", "Amos", "Obadiah",
    "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah", "Haggai", "Zechariah",
    "Malachi", "Matthew", "Mark", "Luke", "John", "Acts", "Romans",
    "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians", "Philippians",
    "Colossians", "1 Thessalonians", "2 Thessalonians", "1 Timothy", "2 Timothy",
    "Titus", "Philemon", "Hebrews", "James", "1 Peter", "2 Peter", "1 John",
    "2 John", "3 John", "Jude", "Revelation",
]

BOOK_ID = {name: i + 1 for i, name in enumerate(CANONICAL_BOOKS)}
ID_BOOK = {i + 1: name for i, name in enumerate(CANONICAL_BOOKS)}

# Abbreviations / spelling variants → canonical name. Numbered books are matched
# both with and without the space (handled in lookup), so "1 cor" and "1cor" work.
_ALIASES = {
    "song of solomon": "Song of Songs", "canticles": "Song of Songs",
    "psalm": "Psalms", "psa": "Psalms", "ps": "Psalms", "pss": "Psalms",
    "gen": "Genesis", "ge": "Genesis", "gn": "Genesis",
    "exo": "Exodus", "exod": "Exodus", "ex": "Exodus",
    "lev": "Leviticus", "lv": "Leviticus",
    "num": "Numbers", "nm": "Numbers", "nb": "Numbers",
    "deut": "Deuteronomy", "deu": "Deuteronomy", "dt": "Deuteronomy",
    "josh": "Joshua", "jos": "Joshua",
    "judg": "Judges", "jdg": "Judges",
    "rut": "Ruth", "ru": "Ruth",
    "1 sam": "1 Samuel", "1 sa": "1 Samuel",
    "2 sam": "2 Samuel", "2 sa": "2 Samuel",
    "1 kgs": "1 Kings", "1 ki": "1 Kings", "1 kin": "1 Kings",
    "2 kgs": "2 Kings", "2 ki": "2 Kings",
    "1 chr": "1 Chronicles", "1 ch": "1 Chronicles",
    "2 chr": "2 Chronicles", "2 ch": "2 Chronicles",
    "ezr": "Ezra", "neh": "Nehemiah", "est": "Esther", "jb": "Job",
    "prov": "Proverbs", "pro": "Proverbs", "prv": "Proverbs", "pr": "Proverbs",
    "eccl": "Ecclesiastes", "ecc": "Ecclesiastes", "qoh": "Ecclesiastes",
    "isa": "Isaiah", "is": "Isaiah",
    "jer": "Jeremiah", "je": "Jeremiah",
    "lam": "Lamentations", "la": "Lamentations",
    "ezek": "Ezekiel", "eze": "Ezekiel", "ezk": "Ezekiel",
    "dan": "Daniel", "dn": "Daniel",
    "hos": "Hosea", "ho": "Hosea",
    "jl": "Joel", "joe": "Joel",
    "am": "Amos", "amo": "Amos",
    "oba": "Obadiah", "ob": "Obadiah",
    "jon": "Jonah", "jnh": "Jonah",
    "mic": "Micah", "mi": "Micah",
    "nah": "Nahum", "na": "Nahum",
    "hab": "Habakkuk", "zep": "Zephaniah", "zeph": "Zephaniah",
    "hag": "Haggai", "hg": "Haggai",
    "zech": "Zechariah", "zec": "Zechariah",
    "mal": "Malachi", "ml": "Malachi",
    "matt": "Matthew", "mat": "Matthew", "mt": "Matthew",
    "mrk": "Mark", "mk": "Mark", "mr": "Mark",
    "luk": "Luke", "lk": "Luke", "lu": "Luke",
    "jn": "John", "joh": "John", "jhn": "John",
    "act": "Acts", "ac": "Acts",
    "rom": "Romans", "ro": "Romans", "rm": "Romans",
    "1 cor": "1 Corinthians", "1 co": "1 Corinthians",
    "2 cor": "2 Corinthians", "2 co": "2 Corinthians",
    "gal": "Galatians", "ga": "Galatians",
    "eph": "Ephesians", "ephes": "Ephesians",
    "phil": "Philippians", "php": "Philippians", "pp": "Philippians",
    "col": "Colossians",
    "1 thess": "1 Thessalonians", "1 thes": "1 Thessalonians", "1 th": "1 Thessalonians",
    "2 thess": "2 Thessalonians", "2 thes": "2 Thessalonians", "2 th": "2 Thessalonians",
    "1 tim": "1 Timothy", "1 ti": "1 Timothy",
    "2 tim": "2 Timothy", "2 ti": "2 Timothy",
    "tit": "Titus", "phlm": "Philemon", "phm": "Philemon", "pm": "Philemon",
    "heb": "Hebrews", "hb": "Hebrews",
    "jas": "James", "jm": "James", "ja": "James",
    "1 pet": "1 Peter", "1 pe": "1 Peter", "1 pt": "1 Peter",
    "2 pet": "2 Peter", "2 pe": "2 Peter", "2 pt": "2 Peter",
    "1 jn": "1 John", "1 jhn": "1 John", "1 jo": "1 John",
    "2 jn": "2 John", "2 jhn": "2 John", "2 jo": "2 John",
    "3 jn": "3 John", "3 jhn": "3 John", "3 jo": "3 John",
    "jud": "Jude", "jde": "Jude",
    "rev": "Revelation", "rv": "Revelation", "apoc": "Revelation",
    "revelations": "Revelation",   # common misspelling (humans and models alike)
}


def _norm(token: str) -> str:
    """Lowercase, drop periods, collapse internal whitespace."""
    return re.sub(r"\s+", " ", token.strip().lower().replace(".", ""))


# Build lookup tables: normalized form → book_id, plus a no-space fallback so
# "1cor" resolves the same as "1 cor".
_INDEX: dict[str, int] = {}
for _name, _bid in BOOK_ID.items():
    _INDEX[_norm(_name)] = _bid
for _alias, _canon in _ALIASES.items():
    _INDEX[_norm(_alias)] = BOOK_ID[_canon]

_INDEX_NOSPACE: dict[str, int] = {k.replace(" ", ""): v for k, v in _INDEX.items()}


class ReferenceError(ValueError):
    """Raised when a reference string cannot be parsed into a book/chapter/verse."""


def lookup_book(token: str) -> int:
    """Resolve a book name or abbreviation to its book_id (1-66)."""
    n = _norm(token)
    if n in _INDEX:
        return _INDEX[n]
    ns = n.replace(" ", "")
    if ns in _INDEX_NOSPACE:
        return _INDEX_NOSPACE[ns]
    raise ReferenceError(f"unknown book: {token!r}")


# "<book> <chapter>:<verse>" with an optional "-<verse>" range tail.
_REF_RE = re.compile(r"^(.*?)[\s.]+(\d+):(\d+)(?:\s*-\s*(\d+))?$")


def parse_reference(ref: str) -> list[tuple[int, int, int]]:
    """
    Parse one reference into its (book_id, chapter, verse) tuples.

    A single verse → one tuple. A continuous range ("Pro 6:16-19") → one tuple
    per verse in the range (caller treats the whole list as one satisfiable entry).
    """
    ref = ref.strip()
    m = _REF_RE.match(ref)
    if not m:
        raise ReferenceError(f"cannot parse reference: {ref!r}")
    book, chap, v1, v2 = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
    book_id = lookup_book(book)
    end = int(v2) if v2 is not None else v1
    if end < v1:
        raise ReferenceError(f"reversed range in reference: {ref!r}")
    return [(book_id, chap, v) for v in range(v1, end + 1)]


def parse_cell(cell: str) -> list[list[tuple[int, int, int]]]:
    """
    Parse a comma-separated CSV cell into a list of *entries*.

    Each entry is the tuple-list from one reference (length 1 for a single verse,
    >1 for a range). Empty / blank cell → []. Unparseable entries raise.
    """
    if not cell or not cell.strip():
        return []
    entries = []
    for piece in cell.split(","):
        piece = piece.strip()
        if piece:
            entries.append(parse_reference(piece))
    return entries


def format_ref(t: tuple[int, int, int]) -> str:
    """(book_id, chapter, verse) → 'Book chapter:verse' for display."""
    return f"{ID_BOOK[t[0]]} {t[1]}:{t[2]}"
