"""Deep learning model #1: BiLSTM with additive attention, trained from scratch.

Architecture::

    tokens -> Embedding(200) -> BiLSTM(2 x 128) -> additive attention pooling
           -> [attention vector ; max-pool] -> Dropout -> Linear -> logit

The attention weights tell which words drove the decision, which the API
re-uses as an explanation.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from ..config import BiLSTMConfig

PAD, UNK = "<pad>", "<unk>"
TOKEN_RE = re.compile(r"[a-z0-9']+|[^\sa-z0-9]")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


class Vocab:
    def __init__(self, itos: list[str]):
        self.itos = itos
        self.stoi = {t: i for i, t in enumerate(itos)}

    @classmethod
    def build(cls, texts: Sequence[str], min_freq: int) -> "Vocab":
        counts = Counter(tok for t in texts for tok in tokenize(t))
        return cls([PAD, UNK] + sorted(t for t, c in counts.items() if c >= min_freq))

    def encode(self, text: str, max_len: int) -> list[int]:
        unk = self.stoi[UNK]
        ids = [self.stoi.get(tok, unk) for tok in tokenize(text)][:max_len]
        return ids or [unk]

    def __len__(self) -> int:
        return len(self.itos)


class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size: int, cfg: BiLSTMConfig):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, cfg.embed_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(cfg.embed_dim, cfg.hidden_dim, batch_first=True, bidirectional=True)
        self.attn = nn.Sequential(nn.Linear(2 * cfg.hidden_dim, cfg.hidden_dim), nn.Tanh(),
                                  nn.Linear(cfg.hidden_dim, 1, bias=False))
        self.dropout = nn.Dropout(cfg.dropout)
        self.out = nn.Linear(4 * cfg.hidden_dim, 1)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor, return_attention: bool = False):
        h, _ = self.lstm(self.emb_dropout(self.embedding(ids)))           # (B, T, 2H)
        scores = self.attn(h).squeeze(-1).masked_fill(~mask, -1e4)        # (B, T)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), h).squeeze(1)           # (B, 2H)
        max_pool = h.masked_fill(~mask.unsqueeze(-1), -1e4).max(dim=1).values
        logit = self.out(self.dropout(torch.cat([context, max_pool], dim=-1))).squeeze(-1)
        return (logit, weights) if return_attention else logit


def collate(batch: list[tuple[list[int], int]]) -> dict[str, torch.Tensor]:
    max_len = max(len(ids) for ids, _ in batch)
    ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, (seq, _) in enumerate(batch):
        ids[i, : len(seq)] = torch.tensor(seq)
    return {"ids": ids, "mask": ids != 0,
            "labels": torch.tensor([y for _, y in batch], dtype=torch.float)}


class BiLSTMClassifier:
    name = "BiLSTM + Attention"
    collate = staticmethod(collate)

    def __init__(self, vocab: Vocab, cfg: BiLSTMConfig = BiLSTMConfig(), device: str | None = None):
        self.cfg = cfg
        self.vocab = vocab
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = BiLSTMAttention(len(vocab), cfg).to(self.device)

    def encode(self, texts: Sequence[str], labels: Sequence[int] | None = None):
        labels = labels if labels is not None else [0] * len(texts)
        return [(self.vocab.encode(t, self.cfg.max_len), int(y)) for t, y in zip(texts, labels)]

    def forward_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.net(batch["ids"].to(self.device), batch["mask"].to(self.device))

    @torch.no_grad()
    def predict_proba(self, texts: Sequence[str], batch_size: int = 256) -> np.ndarray:
        self.net.eval()
        encoded = self.encode(texts)
        probs = []
        for i in range(0, len(encoded), batch_size):
            batch = collate(encoded[i: i + batch_size])
            probs.append(torch.sigmoid(self.forward_batch(batch)).float().cpu().numpy())
        return np.concatenate(probs) if probs else np.zeros(0)

    @torch.no_grad()
    def attention(self, text: str) -> list[tuple[str, float]]:
        self.net.eval()
        batch = collate(self.encode([text]))
        _, weights = self.net(batch["ids"].to(self.device), batch["mask"].to(self.device),
                              return_attention=True)
        tokens = tokenize(text)[: self.cfg.max_len]
        return list(zip(tokens, weights[0, : len(tokens)].cpu().tolist()))

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        # CPU tensors so the weights load on machines without a GPU.
        torch.save({k: v.cpu() for k, v in self.net.state_dict().items()}, directory / "model.pt")
        (directory / "vocab.json").write_text(json.dumps(self.vocab.itos))
        (directory / "config.json").write_text(json.dumps(asdict(self.cfg), indent=2))

    @classmethod
    def load(cls, directory: Path, device: str | None = None) -> "BiLSTMClassifier":
        cfg = BiLSTMConfig(**json.loads((directory / "config.json").read_text()))
        model = cls(Vocab(json.loads((directory / "vocab.json").read_text())), cfg, device)
        model.net.load_state_dict(torch.load(directory / "model.pt", map_location=model.device))
        model.net.eval()
        return model
