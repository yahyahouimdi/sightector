"""Dataset cleaning, splitting and augmentation.

Pipeline (``python -m sightector prepare``):

1. load the raw CSV and map labels ``-1 -> 1 (harassment)``, ``0 -> 0 (clean)``,
   then append the Civil Comments enrichment (threats, identity attacks, hard
   negatives - see :mod:`sightector.data.external`) when it has been downloaded;
2. normalize every message with :func:`sightector.text.normalize`;
3. drop empty / too-short messages;
4. drop exact duplicates **after** normalization (prevents train/test leakage)
   and messages that appear with contradicting labels;
5. stratified train / validation / test split;
6. augment the training split with adversarially obfuscated copies;
7. build an obfuscated copy of the test split for robustness evaluation.
"""
from __future__ import annotations

import json
import random

import pandas as pd
from sklearn.model_selection import train_test_split

from .. import config
from ..text import normalize
from . import external
from ..text.obfuscation import contains_offensive_term, obfuscate_text


def load_raw(path=config.RAW_DATASET) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns={"headline": "text"})
    df["label"] = (df["label"].astype(int) == -1).astype(int)
    return df[["text", "label"]].assign(source="original", category="")


def load_all(cfg: config.DataConfig) -> pd.DataFrame:
    df = load_raw()
    if cfg.use_external and external.PATH.exists():
        df = pd.concat([df, external.load()], ignore_index=True)
    elif cfg.use_external:
        print("note: external data not downloaded - run `python -m sightector download-external`")
    return df


def clean(df: pd.DataFrame, cfg: config.DataConfig) -> tuple[pd.DataFrame, dict]:
    stats: dict[str, int] = {"raw_rows": len(df)}
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str)
    df["clean_text"] = df["text"].map(normalize)

    df = df[df["clean_text"].str.len() >= cfg.min_chars]
    stats["after_empty_filter"] = len(df)

    n_labels = df.groupby("clean_text")["label"].nunique()
    conflicts = n_labels[n_labels > 1].index
    stats["conflicting_texts"] = int(len(conflicts))
    df = df[~df["clean_text"].isin(conflicts)]

    before = len(df)
    df = df.drop_duplicates(subset="clean_text")
    stats["duplicates_removed"] = before - len(df)
    stats["clean_rows"] = len(df)
    stats["harassment"] = int(df["label"].sum())
    stats["clean"] = int((df["label"] == 0).sum())
    stats["by_source"] = df.groupby("source")["label"].agg(["count", "sum"]).rename(
        columns={"sum": "harassment"}).to_dict(orient="index")
    return df.reset_index(drop=True), stats


def split(df: pd.DataFrame, cfg: config.DataConfig, seed: int = config.SEED):
    strata = df["source"] + "_" + df["label"].astype(str)  # keep source and class mix in every split
    train, temp = train_test_split(
        df, test_size=cfg.val_size + cfg.test_size, stratify=strata, random_state=seed)
    rel_test = cfg.test_size / (cfg.val_size + cfg.test_size)
    val, test = train_test_split(temp, test_size=rel_test, stratify=strata.loc[temp.index], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def augment(train: pd.DataFrame, cfg: config.DataConfig, seed: int = config.SEED) -> pd.DataFrame:
    """Add obfuscated copies; the copies are normalized like real inputs would be."""
    rng = random.Random(seed)
    rows = []
    for text, label, source, category in train[["clean_text", "label", "source", "category"]].itertuples(index=False):
        if label == 1 and contains_offensive_term(text):
            for _ in range(cfg.augment_copies_toxic):
                rows.append((obfuscate_text(text, rng), label, source, category))
        elif label == 0 and rng.random() < cfg.augment_clean_fraction:
            rows.append((obfuscate_text(text, rng, offensive_prob=0.9, other_prob=0.15), label, source, category))
    aug = pd.DataFrame(rows, columns=["raw_obfuscated", "label", "source", "category"])
    aug["clean_text"] = aug["raw_obfuscated"].map(normalize)
    aug["text"] = aug["raw_obfuscated"]
    aug["augmented"] = 1
    base = train.assign(augmented=0)
    out = pd.concat([base, aug[base.columns]], ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def obfuscated_test(test: pd.DataFrame, seed: int = config.SEED + 1) -> pd.DataFrame:
    """Every offensive word disguised, 10% of other words noised (different seed)."""
    rng = random.Random(seed)
    obf = test.copy()
    obf["text"] = [obfuscate_text(t, rng, offensive_prob=1.0, other_prob=0.10) for t in test["clean_text"]]
    obf["clean_text"] = obf["text"].map(normalize)
    return obf


def run(cfg: config.DataConfig = config.DataConfig()) -> dict:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df, stats = clean(load_all(cfg), cfg)
    train, val, test = split(df, cfg)
    train_aug = augment(train, cfg)
    test_obf = obfuscated_test(test)

    cols = ["text", "clean_text", "label", "source", "category"]
    train[cols].to_csv(config.PROCESSED_DIR / "train.csv", index=False)
    train_aug[cols + ["augmented"]].to_csv(config.PROCESSED_DIR / "train_augmented.csv", index=False)
    val[cols].to_csv(config.PROCESSED_DIR / "val.csv", index=False)
    test[cols].to_csv(config.PROCESSED_DIR / "test.csv", index=False)
    test_obf[cols].to_csv(config.PROCESSED_DIR / "test_obfuscated.csv", index=False)

    stats.update(train=len(train), train_augmented=len(train_aug), val=len(val), test=len(test))
    (config.PROCESSED_DIR / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def load_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(config.PROCESSED_DIR / f"{name}.csv", keep_default_na=False)
    df["label"] = df["label"].astype(int)
    df["clean_text"] = df["clean_text"].astype(str)
    return df
