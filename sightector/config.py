"""Central configuration: every path and hyper-parameter lives here."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DATASET = DATA_DIR / "raw" / "dataset.csv"
PROCESSED_DIR = DATA_DIR / "processed"
CHALLENGE_SET = DATA_DIR / "challenge_set.csv"

ARTIFACTS_DIR = Path(os.environ.get("SIGHTECTOR_ARTIFACTS", ROOT / "artifacts"))
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
FIGURES_DIR = ROOT / "docs" / "figures"

SEED = 42


@dataclass(frozen=True)
class DataConfig:
    val_size: float = 0.10
    test_size: float = 0.10
    min_chars: int = 2
    use_external: bool = True          # Civil Comments enrichment (threats, identity attacks)
    augment_copies_toxic: int = 1      # obfuscated copies per toxic training sample
    augment_clean_fraction: float = 0.5  # share of clean samples that get a noisy copy


@dataclass(frozen=True)
class BaselineConfig:
    word_ngrams: tuple[int, int] = (1, 2)
    char_ngrams: tuple[int, int] = (2, 5)
    max_features: int = 200_000
    C: float = 4.0


@dataclass(frozen=True)
class BiLSTMConfig:
    max_len: int = 160
    min_freq: int = 2
    embed_dim: int = 200
    hidden_dim: int = 128
    dropout: float = 0.4
    batch_size: int = 64
    lr: float = 2e-3
    epochs: int = 8
    patience: int = 2


@dataclass(frozen=True)
class TransformerConfig:
    model_name: str = "distilroberta-base"
    max_len: int = 160
    batch_size: int = 32
    lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    epochs: int = 4
    patience: int = 2
    grad_clip: float = 1.0
    fp16: bool = True


@dataclass(frozen=True)
class ServingConfig:
    model: str = os.environ.get("SIGHTECTOR_MODEL", "transformer")  # transformer | bilstm | baseline
    chunk_words: int = 120   # long selections are split into windows of this size
    explain_max_words: int = 60
    severity_high: float = 0.90
