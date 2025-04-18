"""Obfuscation-aware text normalization.

Users who harass online deliberately disguise words to slip past filters:
``fu*ck``, ``f u c k``, ``iiidiot``, ``sh!t``, ``1d10t``, ``ѕtupid`` (Cyrillic
``ѕ``), ``fuck  ing``... A plain word-level model sees these as unknown tokens.

:func:`normalize` undoes those tricks *conservatively*: every repair is
validated against the offensive lexicon, so ordinary words are left untouched
(``4you`` stays ``4you``, ``as`` never becomes ``ass``).

The same function runs during training and at inference time, which keeps the
model input distribution identical in both phases.
"""
from __future__ import annotations

import html
import itertools
import re
import unicodedata
from dataclasses import dataclass, field

from . import lexicon

# Look-alike characters from other scripts that NFKC does not fold.
HOMOGLYPHS = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "і": "i", "ѕ": "s", "ԁ": "d", "ј": "j", "к": "k", "м": "m", "т": "t",
    "н": "h", "в": "b", "ı": "i", "ɡ": "g",
    "α": "a", "ο": "o", "ε": "e", "ι": "i", "κ": "k", "ν": "v", "τ": "t",
    "ρ": "p", "υ": "u", "χ": "x",
})

# Characters used as letter substitutes. Some are ambiguous, hence the lists.
LEET: dict[str, tuple[str, ...]] = {
    "0": ("o",), "1": ("i", "l"), "3": ("e",), "4": ("a",), "5": ("s",),
    "7": ("t",), "8": ("b",), "9": ("g",), "@": ("a",), "$": ("s",),
    "!": ("i", "l"), "|": ("i", "l"), "+": ("t",), "€": ("e",), "£": ("l",),
}
# Characters used to mask or split letters inside a word.
MASK_CHARS = set("*#%^~_.-'`\"")
WORD_CHARS = "a-z0-9" + re.escape("".join(LEET) + "".join(MASK_CHARS))

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
MENTION_RE = re.compile(r"@\w+")
ZERO_WIDTH_RE = re.compile(r"[​-‏⁠﻿­]")
TOKEN_RE = re.compile(rf"[{WORD_CHARS}]+|[^\s{WORD_CHARS}]+")
SPACED_LETTERS_RE = re.compile(r"\b(?:[a-z0-9@$!*][\s.\-_*]{1,2}){2,}[a-z0-9@$!*]\b")
ELONGATION_RE = re.compile(r"([a-z])\1{2,}")
TRAILING_PUNCT = ".,!?;:'\"`-_~"
SPLIT_SUFFIXES = ("ing", "in", "er", "ers", "ed", "head", "hole", "face", "ass", "s")


@dataclass
class NormalizationResult:
    text: str
    repairs: list[tuple[str, str]] = field(default_factory=list)


def _fold_unicode(text: str) -> str:
    text = ZERO_WIDTH_RE.sub("", html.unescape(text))
    text = unicodedata.normalize("NFKC", text).lower().translate(HOMOGLYPHS)
    # Strip accents: "stúpid" -> "stupid".
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def _leet_candidates(token: str, limit: int = 32) -> list[str]:
    options = [LEET.get(c, (c,)) for c in token]
    return ["".join(p) for p in itertools.islice(itertools.product(*options), limit)]


def repair_token(token: str) -> str | None:
    """Return the de-obfuscated offensive term for ``token`` or ``None``."""
    has_letters = any(c.isalpha() for c in token)
    if not has_letters:
        return None
    obfuscated = any(c in LEET or c in MASK_CHARS for c in token)
    doubled = re.search(r"([a-z])\1", token) is not None
    if not obfuscated and not doubled:
        return lexicon.lookup(token) if token in lexicon.VARIANTS else None

    for cand in _leet_candidates(token):
        removed = "".join(c for c in cand if c not in MASK_CHARS)
        found = lexicon.lookup_squeezed(removed) if removed else None
        if found:
            return found
        masked = "".join("?" if c in MASK_CHARS else c for c in cand)
        if "?" in masked:
            letters_squeezed = re.sub(r"([a-z])+", r"", masked)
            found = lexicon.lookup_wildcard(masked) or lexicon.lookup_wildcard(letters_squeezed)
            if found:
                return found
    return None


def _join_spaced_letters(text: str, repairs: list[tuple[str, str]]) -> str:
    def _join(match: re.Match[str]) -> str:
        raw = match.group(0)
        joined = re.sub(r"[\s.\-_]", "", raw)
        found = repair_token(joined) or lexicon.lookup_squeezed(joined)
        if found:
            repairs.append((raw, found))
            return found
        return raw
    return SPACED_LETTERS_RE.sub(_join, text)


def _join_split_words(tokens: list[str], repairs: list[tuple[str, str]]) -> list[str]:
    """Re-join words split by spaces: ``fuck ing`` -> ``fucking``, ``ass hole``."""
    vocab = lexicon.lexicon_set()
    out: list[str] = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens):
            first, second = tokens[i], tokens[i + 1]
            found = lexicon.lookup(first + second)
            if found and (first in vocab or second in vocab or second in SPLIT_SUFFIXES):
                repairs.append((f"{first} {second}", found))
                out.append(found)
                i += 2
                continue
        out.append(tokens[i])
        i += 1
    return out


def normalize_verbose(text: str) -> NormalizationResult:
    """Normalize ``text`` and return the list of repairs that were applied."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    repairs: list[tuple[str, str]] = []

    text = _fold_unicode(text)
    text = URL_RE.sub(" url ", text)
    text = MENTION_RE.sub(" @user ", text)
    text = _join_spaced_letters(text, repairs)

    tokens: list[str] = []
    for tok in TOKEN_RE.findall(text):
        if tok == "@user":
            tokens.append(tok)
            continue
        core = tok.strip(TRAILING_PUNCT)
        if not core:
            tokens.append(tok[:3])  # keep punctuation ("!!!" -> "!!!") but bounded
            continue
        repaired = repair_token(core)
        if repaired and repaired != core:
            repairs.append((core, repaired))
            core = repaired
        elif not repaired:
            core = ELONGATION_RE.sub(r"\1\1", core)  # "soooo" -> "soo"
        tokens.append(core)
        tail = tok[len(tok.rstrip(TRAILING_PUNCT)):]
        if tail and tok.rstrip(TRAILING_PUNCT):
            tokens.append(tail[:3])

    tokens = _join_split_words(tokens, repairs)
    return NormalizationResult(" ".join(tokens).strip(), repairs)


def normalize(text: str) -> str:
    """Return the normalized form of ``text`` (see module docstring)."""
    return normalize_verbose(text).text


def find_offensive_terms(normalized_text: str) -> list[str]:
    """Offensive lexicon terms present in an already-normalized text."""
    vocab = lexicon.lexicon_set()
    seen: list[str] = []
    for tok in re.findall(r"[a-z]+", normalized_text):
        if tok in vocab and tok not in seen:
            seen.append(tok)
    return seen
