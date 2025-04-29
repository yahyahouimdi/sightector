"""Deep learning model #2: fine-tuned pre-trained Transformer (DistilRoBERTa).

A pre-trained language model already understands English syntax and word
meaning; fine-tuning adapts it to harassment detection with few labelled
examples. RoBERTa's byte-level BPE tokenizer never produces "unknown" tokens,
so even residual obfuscation that survives normalization stays visible to the
model as sub-word pieces.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..config import TransformerConfig


class TransformerClassifier:
    name = "DistilRoBERTa (fine-tuned)"

    def __init__(self, cfg: TransformerConfig = TransformerConfig(), source: str | Path | None = None,
                 device: str | None = None):
        self.cfg = cfg
        source = str(source or cfg.model_name)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        # A single logit + BCE loss: identical to softmax over 2 classes, simpler threshold tuning.
        self.net = AutoModelForSequenceClassification.from_pretrained(source, num_labels=1).to(self.device)

    def collate(self, batch: list[tuple[str, int]]) -> dict[str, torch.Tensor]:
        texts, labels = zip(*batch)
        enc = self.tokenizer(list(texts), truncation=True, max_length=self.cfg.max_len,
                             padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(labels, dtype=torch.float)
        return dict(enc)

    def encode(self, texts: Sequence[str], labels: Sequence[int] | None = None):
        labels = labels if labels is not None else [0] * len(texts)
        return list(zip(texts, (int(y) for y in labels)))

    def forward_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
        return self.net(**inputs).logits.squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        self.net.eval()
        probs = []
        for i in range(0, len(texts), batch_size):
            batch = self.collate(self.encode(texts[i: i + batch_size]))
            with torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
                logits = self.forward_batch(batch)
            probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        return np.concatenate(probs) if probs else np.zeros(0)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.net.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)
        (directory / "sightector_config.json").write_text(json.dumps(asdict(self.cfg), indent=2))

    @classmethod
    def load(cls, directory: Path, device: str | None = None) -> "TransformerClassifier":
        cfg = TransformerConfig(**json.loads((directory / "sightector_config.json").read_text()))
        model = cls(cfg, source=directory, device=device)
        model.net.eval()
        return model
