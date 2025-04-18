"""Adversarial obfuscation generator.

Used for two things:

* **data augmentation** - training copies of messages whose words are
  disguised the way real users do it, so the model learns to see through them;
* **robustness evaluation** - an obfuscated copy of the test set measures how
  much performance survives an adversary.

Noise is applied to *clean* messages too, so the model cannot learn the
shortcut "weird spelling => harassment".
"""
from __future__ import annotations

import random

from . import lexicon

_LEET_OUT = {"a": "@4", "e": "3", "i": "1!", "o": "0", "s": "$5", "t": "7", "l": "1", "b": "8", "g": "9"}
_HOMOGLYPH_OUT = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "y": "у", "x": "х", "i": "і", "s": "ѕ"}
_MASKS = "*#.-_"


def _leet(word: str, rng: random.Random) -> str:
    chars = list(word)
    idx = [i for i, c in enumerate(chars) if c in _LEET_OUT]
    for i in rng.sample(idx, k=max(1, len(idx) // 2)) if idx else []:
        chars[i] = rng.choice(_LEET_OUT[chars[i]])
    return "".join(chars)


def _mask(word: str, rng: random.Random) -> str:
    if len(word) < 3:
        return word
    i = rng.randrange(1, len(word) - 1)
    return word[:i] + rng.choice("*#") + word[i + 1:]


def _insert(word: str, rng: random.Random) -> str:
    if len(word) < 3:
        return word
    i = rng.randrange(1, len(word) - 1)
    return word[:i] + rng.choice(_MASKS) + word[i:]


def _elongate(word: str, rng: random.Random) -> str:
    vowels = [i for i, c in enumerate(word) if c in "aeiouy"] or list(range(len(word)))
    i = rng.choice(vowels)
    return word[:i] + word[i] * rng.randint(2, 5) + word[i + 1:]


def _space(word: str, rng: random.Random) -> str:
    return rng.choice([" ", ".", "-"]).join(word)


def _homoglyph(word: str, rng: random.Random) -> str:
    return "".join(_HOMOGLYPH_OUT.get(c, c) if rng.random() < 0.5 else c for c in word)


def _caps(word: str, rng: random.Random) -> str:
    return word.upper() if rng.random() < 0.5 else "".join(
        c.upper() if rng.random() < 0.5 else c for c in word)


OPERATIONS = {
    "leet": _leet, "mask": _mask, "insert": _insert, "elongate": _elongate,
    "space": _space, "homoglyph": _homoglyph, "caps": _caps,
}


def obfuscate_word(word: str, rng: random.Random, ops: tuple[str, ...] | None = None) -> str:
    op = rng.choice(ops or tuple(OPERATIONS))
    return OPERATIONS[op](word, rng)


def obfuscate_text(
    text: str,
    rng: random.Random,
    offensive_prob: float = 0.9,
    other_prob: float = 0.08,
    ops: tuple[str, ...] | None = None,
) -> str:
    """Disguise offensive words (with high prob.) and random other words (low prob.)."""
    vocab = lexicon.lexicon_set()
    words = text.split(" ")
    out = []
    for w in words:
        is_offensive = w.lower().strip(".,!?") in vocab
        p = offensive_prob if is_offensive else other_prob
        out.append(obfuscate_word(w, rng, ops) if w.isalpha() and rng.random() < p else w)
    return " ".join(out)


def contains_offensive_term(text: str) -> bool:
    vocab = lexicon.lexicon_set()
    return any(w.strip(".,!?") in vocab for w in text.lower().split())
