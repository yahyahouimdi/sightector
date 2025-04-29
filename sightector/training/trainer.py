"""Generic PyTorch training loop shared by the BiLSTM and the Transformer.

Features: AdamW, linear warm-up/decay schedule, mixed precision (fp16) on GPU,
gradient clipping, per-epoch validation and early stopping on validation
macro-F1 with restoration of the best weights.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..evaluation.metrics import compute_metrics


@dataclass
class TrainResult:
    history: list[dict] = field(default_factory=list)
    best_epoch: int = 0
    best_score: float = -1.0
    seconds: float = 0.0


class Trainer:
    def __init__(self, net: nn.Module, forward: Callable[[dict], torch.Tensor],
                 optimizer: torch.optim.Optimizer, scheduler=None, epochs: int = 3,
                 patience: int = 2, grad_clip: float = 1.0, fp16: bool = True):
        self.net, self.forward = net, forward
        self.optimizer, self.scheduler = optimizer, scheduler
        self.epochs, self.patience, self.grad_clip = epochs, patience, grad_clip
        self.device_type = next(net.parameters()).device.type
        self.use_amp = fp16 and self.device_type == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def _run_epoch(self, loader: DataLoader, train: bool) -> tuple[float, np.ndarray, np.ndarray]:
        self.net.train(train)
        losses, probs, labels = [], [], []
        for batch in tqdm(loader, leave=False, desc="train" if train else "eval"):
            y = batch["labels"].to(next(self.net.parameters()).device)
            with torch.set_grad_enabled(train), torch.autocast(self.device_type, enabled=self.use_amp):
                logits = self.forward(batch)
                loss = self.loss_fn(logits.float(), y)
            if train:
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step()
            losses.append(loss.item())
            probs.append(torch.sigmoid(logits.detach().float()).cpu().numpy())
            labels.append(y.cpu().numpy())
        return float(np.mean(losses)), np.concatenate(probs), np.concatenate(labels)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> TrainResult:
        result, best_state, bad_epochs = TrainResult(), None, 0
        start = time.time()
        for epoch in range(1, self.epochs + 1):
            train_loss, tr_prob, tr_y = self._run_epoch(train_loader, train=True)
            val_loss, va_prob, va_y = self._run_epoch(val_loader, train=False)
            val = compute_metrics(va_y, va_prob)
            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                   "train_f1": compute_metrics(tr_y, tr_prob)["f1"], **{f"val_{k}": v for k, v in val.items()}}
            result.history.append(row)
            print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"val_macro_f1={val['macro_f1']:.4f} val_auc={val['roc_auc']:.4f}")
            if val["macro_f1"] > result.best_score:
                result.best_score, result.best_epoch, bad_epochs = val["macro_f1"], epoch, 0
                best_state = copy.deepcopy(self.net.state_dict())
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    print(f"early stopping at epoch {epoch}")
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        result.seconds = time.time() - start
        return result
