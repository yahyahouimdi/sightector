"""Inference service: raw text in, explained decision out.

    raw text -> normalize (de-obfuscate) -> model -> threshold -> severity
                                       \\-> explanation (lexicon terms + word occlusion)

Long texts (e.g. a whole selected web page) are split into overlapping
windows and the most harassing window decides, so a single insult cannot be
diluted by surrounding paragraphs.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .. import config
from ..models import TextClassifier, load_model
from ..text import find_offensive_terms, normalize_verbose

FALLBACK_ORDER = ("transformer", "bilstm", "baseline")


@dataclass
class Analysis:
    harassment: bool
    score: float
    severity: str                       # none | medium | high
    threshold: float
    model: str
    normalized_text: str
    offensive_terms: list[str] = field(default_factory=list)
    deobfuscated: list[dict] = field(default_factory=list)
    highlights: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class HarassmentDetector:
    def __init__(self, model: TextClassifier, threshold: float = 0.5, kind: str = "",
                 cfg: config.ServingConfig = config.ServingConfig()):
        self.model, self.threshold, self.kind, self.cfg = model, threshold, kind, cfg

    @classmethod
    def load(cls, kind: str | None = None, models_dir: Path = config.MODELS_DIR,
             device: str | None = None) -> "HarassmentDetector":
        """Load ``kind``; if its artifacts are missing fall back to the next available model."""
        cfg = config.ServingConfig()
        wanted = kind or cfg.model
        order = (wanted,) + tuple(k for k in FALLBACK_ORDER if k != wanted)
        for k in order:
            meta_path = models_dir / k / "metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                return cls(load_model(k, models_dir / k, device), meta["threshold"], k, cfg)
        raise FileNotFoundError(f"No trained model found in {models_dir}. Run `python -m sightector train`.")

    # ------------------------------------------------------------------ scoring
    def _windows(self, text: str) -> list[str]:
        words = text.split()
        size = self.cfg.chunk_words
        if len(words) <= size:
            return [text]
        step = size // 2
        return [" ".join(words[i: i + size]) for i in range(0, len(words) - step, step)]

    def score_normalized(self, texts: list[str]) -> np.ndarray:
        """Harassment probability per normalized text (max over its windows)."""
        windows, owner = [], []
        for i, t in enumerate(texts):
            for w in self._windows(t):
                windows.append(w)
                owner.append(i)
        probs = self.model.predict_proba(windows) if windows else np.zeros(0)
        out = np.zeros(len(texts))
        for i, p in zip(owner, probs):
            out[i] = max(out[i], float(p))
        return out

    def _severity(self, score: float) -> str:
        if score < self.threshold:
            return "none"
        return "high" if score >= max(self.cfg.severity_high, self.threshold) else "medium"

    # ------------------------------------------------------------- explanation
    def _occlusion(self, text: str, base: float) -> list[dict]:
        """Importance of each word = drop in log-odds when the word is removed.

        Log-odds instead of probability: for a saturated score (0.998) removing
        one of two insults barely moves the probability but clearly moves the
        logit. Importances are normalized to sum to 1 over the helpful words.
        """
        words = text.split()
        if not 1 < len(words) <= self.cfg.explain_max_words:
            return [{"word": words[0], "importance": 1.0}] if len(words) == 1 else []

        def logit(p):
            p = np.clip(p, 1e-6, 1 - 1e-6)
            return np.log(p / (1 - p))

        variants = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
        drops = logit(base) - logit(self.model.predict_proba(variants))
        positive = np.clip(drops, 0, None)
        if positive.sum() <= 0:
            return []
        share = positive / positive.sum()
        out = [{"word": w, "importance": round(float(s), 3)} for w, s in zip(words, share) if s >= 0.1]
        return sorted(out, key=lambda h: -h["importance"])[:8]

    # -------------------------------------------------------------------- API
    def analyze(self, text: str, explain: bool = True) -> Analysis:
        start = time.perf_counter()
        norm = normalize_verbose(text)
        score = float(self.score_normalized([norm.text])[0]) if norm.text else 0.0
        harassment = score >= self.threshold
        return Analysis(
            harassment=harassment,
            score=round(score, 4),
            severity=self._severity(score),
            threshold=round(self.threshold, 3),
            model=self.model.name,
            normalized_text=norm.text,
            offensive_terms=find_offensive_terms(norm.text),
            deobfuscated=[{"original": o, "normalized": n} for o, n in norm.repairs],
            highlights=self._occlusion(norm.text, score) if explain and harassment else [],
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
        )

    def analyze_batch(self, texts: list[str]) -> list[Analysis]:
        return [self.analyze(t, explain=False) for t in texts]
