"""Report figures (matplotlib, static PNG).

Colours follow one fixed mapping per model across every figure, so a reader
learns "aqua = Transformer" once.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve  # noqa: E402

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MODEL_COLORS = {"legacy": "#9a9893", "baseline": "#2a78d6", "bilstm": "#eb6834", "transformer": "#1baf7a"}
CLASS_COLORS = ("#2a78d6", "#e34948")
SEQ_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fe", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "semibold", "axes.titlesize": 12,
    "axes.titlelocation": "left", "font.size": 10, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
})


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def class_distribution(stats: dict, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    values = [stats["clean"], stats["harassment"]]
    bars = ax.barh(["Clean", "Harassment"], values, color=CLASS_COLORS, height=0.55)
    total = sum(values)
    for bar, v in zip(bars, values):
        ax.text(v + total * 0.01, bar.get_y() + bar.get_height() / 2, f"{v:,}  ({v / total:.0%})",
                va="center", color=INK_2)
    ax.set_xlim(0, max(values) * 1.3)
    ax.grid(axis="y", visible=False)
    ax.set_title("Class balance after cleaning")
    ax.set_xlabel("Messages")
    return _save(fig, path)


def length_histogram(df, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    words = df["clean_text"].str.split().str.len()
    bins = np.logspace(0, np.log10(max(words.max(), 10)), 40)
    for label, name in ((0, "Clean"), (1, "Harassment")):
        ax.hist(words[df["label"] == label], bins=bins, alpha=0.65, color=CLASS_COLORS[label],
                label=name, edgecolor=SURFACE, linewidth=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Words per message (log scale)")
    ax.set_ylabel("Messages")
    ax.set_title("Message length by class")
    ax.legend()
    return _save(fig, path)


def top_terms(counts: list[tuple[str, int]], path: Path) -> Path:
    terms, values = zip(*counts[::-1])
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.barh(terms, values, color=CLASS_COLORS[1], height=0.6)
    ax.grid(axis="y", visible=False)
    ax.set_title("Most frequent offensive terms (harassment class)")
    ax.set_xlabel("Messages containing the term")
    return _save(fig, path)


def confusion(y_true, y_pred, title: str, path: Path) -> Path:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(cm, cmap=SEQ_CMAP)
    ax.grid(False)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=12,
                color="white" if v > cm.max() * 0.55 else INK)
    ax.set_xticks([0, 1], ["Clean", "Harassment"])
    ax.set_yticks([0, 1], ["Clean", "Harassment"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10.5)
    return _save(fig, path)


def roc_pr_curves(curves: dict[str, tuple], names: dict[str, str], path: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for kind, (y, prob) in curves.items():
        fpr, tpr, _ = roc_curve(y, prob)
        prec, rec, _ = precision_recall_curve(y, prob)
        ax1.plot(fpr, tpr, lw=2, color=MODEL_COLORS[kind], label=names[kind])
        ax2.plot(rec, prec, lw=2, color=MODEL_COLORS[kind], label=names[kind])
    ax1.plot([0, 1], [0, 1], ls="--", lw=1, color=INK_2)
    ax1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (test set)")
    ax2.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve (test set)")
    ax1.legend(loc="lower right")
    ax2.legend(loc="lower left")
    return _save(fig, path)


def grouped_bars(table: dict[str, dict[str, float]], groups: list[str], group_labels: list[str],
                 names: dict[str, str], title: str, ylabel: str, path: Path, ylim=(0, 1.0),
                 note: str = "") -> Path:
    kinds = list(table)
    x = np.arange(len(groups))
    width = 0.8 / len(kinds)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, kind in enumerate(kinds):
        vals = [table[kind].get(g, np.nan) for g in groups]
        bars = ax.bar(x + (i - (len(kinds) - 1) / 2) * width, vals, width * 0.92,
                      color=MODEL_COLORS.get(kind, INK_2), label=names[kind])
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center",
                        fontsize=8, color=INK_2)
    ax.set_xticks(x, group_labels)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(kinds))
    if note:
        fig.text(0.01, -0.17, note, fontsize=8, color=INK_2, style="italic")
    return _save(fig, path)


def category_heatmap(table: dict[str, dict[str, float]], names: dict[str, str], title: str,
                     path: Path) -> Path:
    kinds = list(table)
    cats = sorted({c for d in table.values() for c in d})
    data = np.array([[table[k].get(c, np.nan) for c in cats] for k in kinds])
    fig, ax = plt.subplots(figsize=(10, 0.65 * len(kinds) + 1.4))
    ax.imshow(data, cmap=SEQ_CMAP, vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    for (i, j), v in np.ndenumerate(data):
        ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9,
                color="white" if v > 0.6 else INK)
    ax.set_xticks(range(len(cats)), [c.replace(" ", "\n") for c in cats])
    ax.set_yticks(range(len(kinds)), [names[k] for k in kinds])
    ax.tick_params(length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title(title)
    return _save(fig, path)


def learning_curves(histories: dict[str, list[dict]], names: dict[str, str], path: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for kind, hist in histories.items():
        ep = [h["epoch"] for h in hist]
        c = MODEL_COLORS[kind]
        ax1.plot(ep, [h["train_loss"] for h in hist], lw=2, color=c, ls="--", marker="o", ms=4)
        ax1.plot(ep, [h["val_loss"] for h in hist], lw=2, color=c, marker="o", ms=4, label=names[kind])
        ax2.plot(ep, [h["val_macro_f1"] for h in hist], lw=2, color=c, marker="o", ms=4, label=names[kind])
    ax1.set(xlabel="Epoch", ylabel="BCE loss", title="Loss (dashed = train, solid = validation)")
    ax2.set(xlabel="Epoch", ylabel="Macro-F1", title="Validation macro-F1")
    for ax in (ax1, ax2):
        ax.legend()
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    return _save(fig, path)


def ablation(rows: list[dict], path: Path) -> Path:
    labels = [r["setting"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(rows))
    for offset, key, color, name in ((-0.18, "test_obfuscated", "#2a78d6", "Obfuscated test set"),
                                     (0.18, "challenge", "#eb6834", "Challenge set")):
        vals = [r[key] for r in rows]
        bars = ax.barh(y + offset, vals, 0.34, color=color, label=name)
        for b, v in zip(bars, vals):
            ax.text(v + 0.005, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(min(min(r["test_obfuscated"], r["challenge"]) for r in rows) - 0.1, 1.04)
    ax.set_xlabel("Macro-F1")
    ax.set_title("Ablation (DistilRoBERTa): normalizer x augmentation")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    return _save(fig, path)
