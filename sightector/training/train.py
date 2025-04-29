"""Training entry points for the three models.

Each function trains a model, tunes the decision threshold on the validation
split and writes ``<artifacts>/models/<kind>/`` with the weights plus a
``metadata.json`` (threshold, validation metrics, learning curves, timing).
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .. import config
from ..data.prepare import load_split
from ..evaluation.metrics import best_threshold, compute_metrics
from .trainer import Trainer


def set_seed(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_data(augmented: bool):
    train = load_split("train_augmented" if augmented else "train")
    return train, load_split("val")


def _finalize(model, kind: str, val, out_dir: Path, extra: dict) -> dict:
    prob = model.predict_proba(val["clean_text"].tolist())
    threshold = best_threshold(val["label"], prob)
    meta = {"kind": kind, "name": model.name, "threshold": threshold,
            "val_metrics": compute_metrics(val["label"], prob, threshold), **extra}
    model.save(out_dir)
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[{kind}] saved to {out_dir} | threshold={threshold:.2f} | "
          f"val macro-F1={meta['val_metrics']['macro_f1']:.4f}")
    return meta


def train_baseline(augmented: bool = True, out_dir: Path | None = None) -> dict:
    from ..models.baseline import TfidfLogReg

    set_seed()
    train, val = _load_data(augmented)
    start = time.time()
    model = TfidfLogReg().fit(train["clean_text"], train["label"])
    return _finalize(model, "baseline", val, out_dir or config.MODELS_DIR / "baseline",
                     {"augmented": augmented, "train_seconds": time.time() - start})


def _torch_fit(model, train, val, batch_size, epochs, patience, optimizer, scheduler_fn=None,
               grad_clip=1.0, fp16=True):
    train_loader = DataLoader(model.encode(train["clean_text"].tolist(), train["label"].tolist()),
                              batch_size=batch_size, shuffle=True, collate_fn=model.collate)
    val_loader = DataLoader(model.encode(val["clean_text"].tolist(), val["label"].tolist()),
                            batch_size=batch_size * 2, collate_fn=model.collate)
    scheduler = scheduler_fn(len(train_loader) * epochs) if scheduler_fn else None
    trainer = Trainer(model.net, model.forward_batch, optimizer, scheduler, epochs=epochs,
                      patience=patience, grad_clip=grad_clip, fp16=fp16)
    return trainer.fit(train_loader, val_loader)


def train_bilstm(augmented: bool = True, out_dir: Path | None = None,
                 cfg: config.BiLSTMConfig = config.BiLSTMConfig()) -> dict:
    from ..models.bilstm import BiLSTMClassifier, Vocab

    set_seed()
    train, val = _load_data(augmented)
    model = BiLSTMClassifier(Vocab.build(train["clean_text"], cfg.min_freq), cfg)
    optimizer = torch.optim.AdamW(model.net.parameters(), lr=cfg.lr, weight_decay=1e-4)
    result = _torch_fit(model, train, val, cfg.batch_size, cfg.epochs, cfg.patience, optimizer)
    return _finalize(model, "bilstm", val, out_dir or config.MODELS_DIR / "bilstm", {
        "augmented": augmented, "vocab_size": len(model.vocab), "history": result.history,
        "best_epoch": result.best_epoch, "train_seconds": result.seconds,
        "parameters": sum(p.numel() for p in model.net.parameters())})


def train_transformer(augmented: bool = True, out_dir: Path | None = None,
                      cfg: config.TransformerConfig = config.TransformerConfig()) -> dict:
    from ..models.transformer import TransformerClassifier

    set_seed()
    train, val = _load_data(augmented)
    model = TransformerClassifier(cfg)
    no_decay = ("bias", "LayerNorm.weight")
    params = [
        {"params": [p for n, p in model.net.named_parameters() if not n.endswith(no_decay)],
         "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in model.net.named_parameters() if n.endswith(no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr)

    def scheduler_fn(total_steps: int):
        return get_linear_schedule_with_warmup(optimizer, int(cfg.warmup_ratio * total_steps), total_steps)

    result = _torch_fit(model, train, val, cfg.batch_size, cfg.epochs, cfg.patience, optimizer,
                        scheduler_fn, cfg.grad_clip, cfg.fp16)
    return _finalize(model, "transformer", val, out_dir or config.MODELS_DIR / "transformer", {
        "augmented": augmented, "base_model": cfg.model_name, "history": result.history,
        "best_epoch": result.best_epoch, "train_seconds": result.seconds,
        "parameters": sum(p.numel() for p in model.net.parameters())})


TRAINERS = {"baseline": train_baseline, "bilstm": train_bilstm, "transformer": train_transformer}
