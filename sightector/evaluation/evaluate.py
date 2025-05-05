"""Benchmark every trained model and produce the report figures/tables.

Evaluation sets
---------------
* ``test``            - held-out 10 % of the cleaned dataset (original + enrichment);
* ``test_original``   - the part of ``test`` that comes from the original dataset
                        (like-for-like comparison with the v1 system);
* ``test_obfuscated`` - the same messages with every offensive word disguised;
* ``challenge``       - 78 hand-written messages (independent of the dataset)
                        covering obfuscation, threats, sexism, tricky clean text.

The legacy v1 system (word TF-IDF + LinearSVC, served with the IDF re-fit bug)
is scored on the same sets for a before/after comparison.
"""
from __future__ import annotations

import json
import pickle
import re
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from .. import config
from ..data.prepare import load_split
from ..models import load_model
from ..text import find_offensive_terms, normalize
from . import plots
from .metrics import compute_metrics

NAMES = {"legacy": "Legacy v1 (TF-IDF + LinearSVC)", "baseline": "TF-IDF + LogReg",
         "bilstm": "BiLSTM + Attention", "transformer": "DistilRoBERTa"}
SETS = ["test", "test_original", "test_obfuscated", "challenge"]
SET_LABELS = ["Test set", "Test (original data)", "Obfuscated test set", "Challenge set"]
LEGACY_DIR = config.ROOT / "legacy" / "DetectionModelServer"


def load_eval_sets() -> dict[str, pd.DataFrame]:
    challenge = pd.read_csv(config.CHALLENGE_SET)
    challenge["clean_text"] = challenge["text"].map(normalize)
    test = load_split("test")
    return {"test": test, "test_original": test[test["source"] == "original"].reset_index(drop=True),
            "test_obfuscated": load_split("test_obfuscated"), "challenge": challenge}


class LegacyModel:
    """Exact reproduction of the v1 server: vectorizer re-fitted on each input."""

    name = NAMES["legacy"]

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        stop = (LEGACY_DIR / "stopwords.txt").read_text().splitlines()
        vocab = pickle.load(open(LEGACY_DIR / "tfidfvectoizer.pkl", "rb"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = pickle.load(open(LEGACY_DIR / "LinearSVCTuned.pkl", "rb"))
        self.make_vec = lambda: TfidfVectorizer(stop_words=stop, lowercase=True, vocabulary=vocab)

    def predict_proba(self, texts):
        preds = []
        for t in texts:
            try:
                preds.append(int(self.model.predict(self.make_vec().fit_transform([t]))[0]))
            except ValueError:  # empty vocabulary after stop-word removal
                preds.append(0)
        return np.array(preds, dtype=float)


def available_models() -> dict[str, tuple[object, float]]:
    out = {}
    if (LEGACY_DIR / "LinearSVCTuned.pkl").exists():
        out["legacy"] = (LegacyModel(), 0.5)
    for kind in ("baseline", "bilstm", "transformer"):
        meta = config.MODELS_DIR / kind / "metadata.json"
        if meta.exists():
            out[kind] = (load_model(kind), json.loads(meta.read_text())["threshold"])
    return out


def run_eda() -> None:
    stats = json.loads((config.PROCESSED_DIR / "stats.json").read_text())
    df = pd.concat([load_split(s) for s in ("train", "val", "test")])
    plots.class_distribution(stats, config.FIGURES_DIR / "eda_class_balance.png")
    plots.length_histogram(df, config.FIGURES_DIR / "eda_length.png")
    counts = Counter(t for text in df.loc[df.label == 1, "clean_text"] for t in find_offensive_terms(text))
    plots.top_terms(counts.most_common(15), config.FIGURES_DIR / "eda_top_terms.png")


def run_benchmark() -> pd.DataFrame:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    sets = load_eval_sets()
    models = available_models()
    rows, curves, f1_table, per_category, errors = [], {}, {}, {}, None

    for kind, (model, threshold) in models.items():
        f1_table[kind] = {}
        for set_name in SETS:
            df = sets[set_name]
            # The legacy server received raw text; new models receive normalized text.
            inputs = df["text"] if kind == "legacy" else df["clean_text"]
            prob = model.predict_proba(inputs.tolist())
            m = compute_metrics(df["label"], prob, threshold)
            if kind == "legacy":
                m["roc_auc"] = m["pr_auc"] = float("nan")
            rows.append({"model": NAMES[kind], "kind": kind, "set": set_name, **m})
            f1_table[kind][set_name] = m["macro_f1"]
            pred = (prob >= threshold).astype(int)
            if set_name == "test" and kind != "legacy":
                curves[kind] = (df["label"].values, prob)
                plots.confusion(df["label"], pred, f"{NAMES[kind]} - test set",
                                config.FIGURES_DIR / f"confusion_{kind}.png")
                if kind == "transformer":
                    errors = df.assign(prob=prob, pred=pred).query("pred != label")
            if set_name == "challenge":
                per_category[kind] = (df.assign(correct=pred == df["label"])
                                      .groupby("category")["correct"].mean().to_dict())
                df.assign(pred=pred, prob=prob.round(3)).to_csv(
                    config.REPORTS_DIR / f"challenge_predictions_{kind}.csv", index=False)

    results = pd.DataFrame(rows)
    results.to_csv(config.REPORTS_DIR / "benchmark.csv", index=False)
    (config.REPORTS_DIR / "challenge_by_category.json").write_text(json.dumps(per_category, indent=2))
    if errors is not None:
        errors.sort_values("prob").to_csv(config.REPORTS_DIR / "transformer_errors.csv", index=False)

    names = {k: NAMES[k] for k in models}
    note = ("Legacy v1 on 'Test (original data)' is inflated: v1 was trained on the whole original dataset, "
            "so these test rows were part of its training data." if "legacy" in models else "")
    plots.grouped_bars(f1_table, SETS, SET_LABELS, names, "Macro-F1 by model and evaluation set",
                       "Macro-F1", config.FIGURES_DIR / "benchmark_macro_f1.png", ylim=(0, 1.08), note=note)
    plots.category_heatmap(per_category, names, "Challenge set accuracy by category",
                           config.FIGURES_DIR / "challenge_by_category.png")
    if curves:
        plots.roc_pr_curves(curves, names, config.FIGURES_DIR / "roc_pr_curves.png")
    histories = {}
    for kind in ("bilstm", "transformer"):
        meta = config.MODELS_DIR / kind / "metadata.json"
        if meta.exists():
            histories[kind] = json.loads(meta.read_text())["history"]
    if histories:
        plots.learning_curves(histories, names, config.FIGURES_DIR / "learning_curves.png")
    write_markdown_table(results)
    return results


def run_ablation() -> list[dict]:
    """Transformer with/without augmentation x with/without the normalizer."""
    noaug_dir = config.MODELS_DIR / "ablation_transformer_noaug"
    if not (noaug_dir / "metadata.json").exists():
        raise FileNotFoundError("Train it first: python -m sightector train transformer --no-augment "
                                f"--out {noaug_dir}")
    sets = load_eval_sets()
    rows = []
    for aug_name, directory in (("no augmentation", noaug_dir), ("augmentation", config.MODELS_DIR / "transformer")):
        model = load_model("transformer", directory)
        threshold = json.loads((directory / "metadata.json").read_text())["threshold"]
        for norm in (False, True):
            row = {"setting": f"{'normalizer' if norm else 'raw input'} + {aug_name}"}
            for set_name in ("test", "test_obfuscated", "challenge"):
                df = sets[set_name]
                inputs = df["clean_text"] if norm else df["text"].str.lower()
                prob = model.predict_proba(inputs.tolist())
                row[set_name] = compute_metrics(df["label"], prob, threshold)["macro_f1"]
            rows.append(row)
    (config.REPORTS_DIR / "ablation.json").write_text(json.dumps(rows, indent=2))
    plots.ablation(rows, config.FIGURES_DIR / "ablation.png")
    return rows


def write_markdown_table(results: pd.DataFrame) -> None:
    lines = []
    for set_name, label in zip(SETS, SET_LABELS):
        lines += [f"\n### {label}\n", "| Model | Accuracy | Precision | Recall | F1 | Macro-F1 | ROC-AUC |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in results[results.set == set_name].iterrows():
            auc = "-" if np.isnan(r.roc_auc) else f"{r.roc_auc:.3f}"
            lines.append(f"| {r.model} | {r.accuracy:.3f} | {r.precision:.3f} | {r.recall:.3f} | "
                         f"{r.f1:.3f} | {r.macro_f1:.3f} | {auc} |")
    text = "\n".join(lines)
    (config.REPORTS_DIR / "benchmark.md").write_text(re.sub(r"\n{3,}", "\n\n", text).strip() + "\n")
