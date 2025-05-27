"""Draw the architecture diagrams used in the README, report and slides.

Usage: python scripts/make_diagrams.py   ->  docs/figures/architecture.png, pipeline.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "figures"
NAVY, BLUE, RED, INK_2, SURFACE = "#0c2547", "#0c6a9d", "#cc163f", "#52514e", "#fcfcfb"
TINTS = {"navy": "#e3e9f2", "blue": "#ddeef8", "red": "#fbe3e8", "gray": "#efefec"}


def box(ax, x, y, w, h, title, sub="", tint="blue", edge=BLUE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=TINTS[tint], ec=edge, lw=1.4))
    lines = sub.count("\n") + 1 if sub else 0
    ax.text(x + w / 2, y + h / 2 + 0.09 * lines, title, ha="center", va="center",
            fontsize=10, fontweight="bold", color=NAVY)
    if sub:
        ax.text(x + w / 2, y + h / 2 + 0.09 * lines - 0.2, sub, ha="center", va="top", fontsize=8.2,
                color=INK_2, linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, label="", color=INK_2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                 lw=1.4, color=color, shrinkA=2, shrinkB=2))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.1, label, ha="center", fontsize=8, color=INK_2)


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(SURFACE)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def architecture() -> None:
    fig, ax = canvas(12, 4.6)
    ax.text(0.2, 4.3, "System architecture", fontsize=13, fontweight="semibold", color=NAVY)
    box(ax, 0.2, 1.6, 2.2, 1.3, "Browser extension", "Chrome MV3 popup\nselected text / typed text", "red", RED)
    box(ax, 0.2, 0.15, 2.2, 1.0, "Web dashboard", "live demo  ·  /?q=", "red", RED)
    box(ax, 3.2, 1.0, 2.2, 1.6, "Flask REST API", "server/\nPOST /api/analyze\nGET /api/health", "navy", NAVY)
    box(ax, 6.2, 2.45, 2.5, 1.2, "Normalizer", "de-obfuscation\nleet · masks · spacing · Unicode")
    box(ax, 6.2, 0.95, 2.5, 1.2, "DistilRoBERTa", "fine-tuned classifier\n(fallback: BiLSTM / TF-IDF)")
    box(ax, 9.4, 2.45, 2.4, 1.2, "Explainer", "lexicon terms\nword occlusion")
    box(ax, 9.4, 0.95, 2.4, 1.2, "Decision", "tuned threshold\nseverity none/medium/high")
    arrow(ax, 2.4, 2.25, 3.2, 1.95, "JSON")
    arrow(ax, 2.4, 0.65, 3.2, 1.35)
    arrow(ax, 5.4, 2.0, 6.2, 3.0)
    arrow(ax, 7.45, 2.45, 7.45, 2.15)
    arrow(ax, 8.7, 1.55, 9.4, 1.55)
    arrow(ax, 8.7, 3.05, 9.4, 3.05)
    # the result travels back below the model boxes
    ax.plot([10.6, 10.6, 4.3], [0.95, 0.55, 0.55], color=BLUE, lw=1.4)
    arrow(ax, 4.3, 0.55, 4.3, 1.0, color=BLUE)
    ax.text(7.45, 0.62, "explained JSON result", ha="center", fontsize=8, color=BLUE)
    ax.text(6.2, 3.85, "sightector/inference · HarassmentDetector", fontsize=8.5, color=INK_2, style="italic")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "architecture.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def pipeline() -> None:
    fig, ax = canvas(12.5, 2.6)
    ax.text(0.15, 2.3, "Training pipeline", fontsize=13, fontweight="semibold", color=NAVY)
    steps = [("Raw data", "16.5k messages\n+ Civil Comments", "gray", INK_2),
             ("Cleaning", "normalize · dedupe\nconflicts removed", "blue", BLUE),
             ("Split", "80 / 10 / 10\nstratified", "blue", BLUE),
             ("Augmentation", "adversarial\nobfuscated copies", "red", RED),
             ("Training", "TF-IDF · BiLSTM\nDistilRoBERTa", "navy", NAVY),
             ("Threshold", "tuned on\nvalidation", "blue", BLUE),
             ("Evaluation", "test · obfuscated\nchallenge · ablation", "navy", NAVY)]
    w, gap = 1.6, 0.18
    for i, (title, sub, tint, edge) in enumerate(steps):
        x = 0.15 + i * (w + gap)
        box(ax, x, 0.45, w, 1.35, title, sub, tint, edge)
        if i:
            arrow(ax, x - gap, 1.12, x, 1.12)
    fig.savefig(OUT / "pipeline.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    pipeline()
    print("saved", OUT / "architecture.png", OUT / "pipeline.png")
