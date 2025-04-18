"""Offensive-term lexicon.

The lexicon is used in two places only:

* the normalizer, to *validate* de-obfuscation repairs (``f*ck`` -> ``fuck``),
  so an ordinary word is never rewritten into something it is not;
* the explainer, to report which offensive terms were found in a message.

The harassment decision itself is always taken by the trained model.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

LEXICON_PATH = Path(__file__).resolve().parent.parent / "resources" / "offensive_lexicon.txt"

# Common abbreviations / phonetic spellings that cannot be recovered by
# character rules alone. Values may contain several words.
VARIANTS: dict[str, str] = {
    "fck": "fuck", "fk": "fuck", "fuk": "fuck", "fuq": "fuck", "phuck": "fuck",
    "fcking": "fucking", "fking": "fucking", "fkn": "fucking", "fuking": "fucking",
    "fukin": "fucking", "effing": "fucking", "fcker": "fucker", "fuker": "fucker",
    "stfu": "shut the fuck up", "gtfo": "get the fuck out", "wtf": "what the fuck",
    "mf": "motherfucker", "mofo": "motherfucker", "mfer": "motherfucker",
    "biatch": "bitch", "bich": "bitch", "btch": "bitch", "biotch": "bitch",
    "sht": "shit", "shyt": "shit", "shiet": "shit",
    "azz": "ass", "arsehole": "asshole", "a-hole": "asshole", "ahole": "asshole",
    "idot": "idiot", "idjit": "idiot", "idiyot": "idiot",
    "stoopid": "stupid", "stupd": "stupid", "stpid": "stupid",
    "kys": "kill yourself", "dik": "dick", "nigg": "nigger",
    "retart": "retard", "tard": "retard", "fagot": "faggot", "fagg": "fag",
    "cnt": "cunt", "hore": "whore", "slutt": "slut",
}


def squeeze(word: str) -> str:
    """Collapse every run of the same character to one: ``iiidiooot`` -> ``idiot``."""
    return re.sub(r"(.)\1+", r"\1", word)


@lru_cache(maxsize=1)
def load_lexicon() -> tuple[str, ...]:
    """Terms in file order (the order is used as priority to break ties)."""
    terms: list[str] = []
    for line in LEXICON_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip().lower()
        if line and not line.startswith("#") and line not in terms:
            terms.append(line)
    return tuple(terms)


@lru_cache(maxsize=1)
def lexicon_set() -> frozenset[str]:
    return frozenset(load_lexicon())


@lru_cache(maxsize=1)
def squeezed_index() -> dict[str, str]:
    """Map ``squeeze(term)`` -> term, keeping the highest-priority term."""
    index: dict[str, str] = {}
    for term in load_lexicon():
        index.setdefault(squeeze(term), term)
    return index


def lookup(candidate: str) -> str | None:
    """Return the canonical offensive term for ``candidate`` or ``None``."""
    if candidate in lexicon_set():
        return candidate
    if candidate in VARIANTS:
        return VARIANTS[candidate]
    return None


def lookup_squeezed(candidate: str) -> str | None:
    """Like :func:`lookup` but tolerant to letter elongation (``fuuuck``)."""
    found = lookup(candidate)
    if found:
        return found
    key = squeeze(candidate)
    return squeezed_index().get(key) or VARIANTS.get(key)


def lookup_wildcard(pattern: str, wildcard: str = "?") -> str | None:
    """Match a masked token such as ``f?ck`` against the lexicon (first wins)."""
    if pattern.count(wildcard) == len(pattern) or sum(c.isalpha() for c in pattern) < 1:
        return None
    regex = re.compile("^" + re.escape(pattern).replace(re.escape(wildcard), "[a-z]") + "$")
    for term in load_lexicon():
        if regex.match(term):
            return term
    return None
