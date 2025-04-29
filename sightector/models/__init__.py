"""Model layer.

Every model exposes the same small interface (:class:`TextClassifier`) so the
training, evaluation and serving code never depends on a concrete model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import numpy as np


class TextClassifier(Protocol):
    name: str

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Harassment probability for each *normalized* text, shape ``(n,)``."""

    def save(self, directory: Path) -> None: ...


def load_model(kind: str, directory: Path | None = None, device: str | None = None) -> TextClassifier:
    """Factory used by evaluation and serving: ``kind`` in {baseline, bilstm, transformer}."""
    from .. import config

    directory = directory or config.MODELS_DIR / kind
    if kind == "baseline":
        from .baseline import TfidfLogReg
        return TfidfLogReg.load(directory)
    if kind == "bilstm":
        from .bilstm import BiLSTMClassifier
        return BiLSTMClassifier.load(directory, device=device)
    if kind == "transformer":
        from .transformer import TransformerClassifier
        return TransformerClassifier.load(directory, device=device)
    raise ValueError(f"Unknown model kind: {kind!r}")
