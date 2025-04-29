"""Classical baseline: TF-IDF (word + character n-grams) + Logistic Regression.

Character n-grams (2-5) make the baseline partially robust to spelling noise,
which the original word-only TF-IDF + LinearSVC was not.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from ..config import BaselineConfig


class TfidfLogReg:
    name = "TF-IDF + LogReg"

    def __init__(self, cfg: BaselineConfig = BaselineConfig(), pipeline: Pipeline | None = None):
        self.cfg = cfg
        self.pipeline = pipeline or Pipeline([
            ("features", FeatureUnion([
                ("word", TfidfVectorizer(ngram_range=cfg.word_ngrams, sublinear_tf=True,
                                         min_df=2, max_features=cfg.max_features)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=cfg.char_ngrams,
                                         sublinear_tf=True, min_df=3, max_features=cfg.max_features)),
            ])),
            ("clf", LogisticRegression(C=cfg.C, max_iter=2000, class_weight="balanced")),
        ])

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> "TfidfLogReg":
        self.pipeline.fit(list(texts), list(labels))
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return self.pipeline.predict_proba(list(texts))[:, 1]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump({"cfg": self.cfg, "pipeline": self.pipeline}, directory / "model.joblib")

    @classmethod
    def load(cls, directory: Path) -> "TfidfLogReg":
        state = joblib.load(directory / "model.joblib")
        return cls(state["cfg"], state["pipeline"])
