"""Data enrichment with Civil Comments (Jigsaw, CC0 licence).

Error analysis of the first models showed the original dataset - mostly
Wikipedia insults - contains almost no *threats* or *identity attacks*, so
"i know where you live and i will hurt you" was classified as clean by every
model. We add real, human-annotated comments for the missing categories, plus
"hard negatives": clean comments that use violent or offensive-looking words
in a harmless way ("this exam is killing me"), so the model learns context
instead of keywords.

Download (≈190 MB, done once):  python -m sightector download-external
"""
from __future__ import annotations

import urllib.request

import pandas as pd

from .. import config

URL = "https://huggingface.co/api/datasets/google/civil_comments/parquet/default/train/0.parquet"
PATH = config.DATA_DIR / "external" / "civil_comments_train_0.parquet"

# Annotator-agreement thresholds: >= 0.5 means at least half the raters agreed.
TOXIC_SELECTION = {          # column -> number of comments sampled
    "threat": 2000,
    "identity_attack": 1500,
    "sexual_explicit": 1000,
    "insult": 500,
}
HARD_NEGATIVE_WORDS = r"\b(?:kill|killing|killed|die|dead|shoot|beat|hurt|attack|destroy|fight|hate|stupid|crazy|gun)\b"
N_HARD_NEGATIVES = 2500
N_RANDOM_CLEAN = 2000
MAX_CHARS = 500


def download(path=PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL} -> {path}")
    urllib.request.urlretrieve(URL, path)


def load(path=PATH, seed: int = config.SEED) -> pd.DataFrame:
    """Return the selected comments as ``text, label, source, category``."""
    df = pd.read_parquet(path)
    df = df[df["text"].str.len().between(10, MAX_CHARS)]
    parts = []
    taken = pd.Index([])
    for column, n in TOXIC_SELECTION.items():
        pool = df[(df[column] >= 0.5) & ~df.index.isin(taken)]
        sample = pool.sample(min(n, len(pool)), random_state=seed)
        taken = taken.union(sample.index)
        parts.append(sample.assign(label=1, category=column))

    clean = df[(df["toxicity"] < 0.1) & ~df.index.isin(taken)]
    hard = clean[clean["text"].str.contains(HARD_NEGATIVE_WORDS, case=False, regex=True)]
    hard = hard.sample(min(N_HARD_NEGATIVES, len(hard)), random_state=seed)
    rest = clean.drop(hard.index).sample(N_RANDOM_CLEAN, random_state=seed)
    parts += [hard.assign(label=0, category="hard_negative"), rest.assign(label=0, category="clean")]

    out = pd.concat(parts)[["text", "label", "category"]].assign(source="civil_comments")
    return out.reset_index(drop=True)
