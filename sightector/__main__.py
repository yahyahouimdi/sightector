"""Command-line interface.

    python -m sightector download-external            # Civil Comments enrichment (~190 MB, once)
    python -m sightector prepare                      # clean + split + augment the dataset
    python -m sightector train all                    # baseline, bilstm, transformer
    python -m sightector train transformer --no-augment --out artifacts/models/ablation_transformer_noaug
    python -m sightector evaluate                     # benchmark + figures (+ ablation if available)
    python -m sightector predict "you are a fu*ck idiot"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sightector", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download-external", help="download the Civil Comments enrichment data")
    sub.add_parser("prepare", help="clean, split and augment the raw dataset")
    train = sub.add_parser("train", help="train one model or all of them")
    train.add_argument("model", choices=["baseline", "bilstm", "transformer", "all"])
    train.add_argument("--no-augment", action="store_true", help="train without obfuscation augmentation")
    train.add_argument("--out", type=Path, default=None, help="output directory")
    sub.add_parser("evaluate", help="benchmark all trained models and draw the figures")
    predict = sub.add_parser("predict", help="analyze a message")
    predict.add_argument("text")
    predict.add_argument("--model", default=None)
    args = parser.parse_args(argv)

    if args.command == "download-external":
        from .data.external import download
        download()
    elif args.command == "prepare":
        from .data.prepare import run
        print(json.dumps(run(), indent=2))
    elif args.command == "train":
        from .training.train import TRAINERS
        kinds = list(TRAINERS) if args.model == "all" else [args.model]
        for kind in kinds:
            TRAINERS[kind](augmented=not args.no_augment, out_dir=args.out)
    elif args.command == "evaluate":
        from .evaluation import evaluate
        evaluate.run_eda()
        print(evaluate.run_benchmark().to_string(index=False))
        try:
            print(json.dumps(evaluate.run_ablation(), indent=2))
        except FileNotFoundError as exc:
            print(f"(ablation skipped) {exc}")
    elif args.command == "predict":
        from .inference import HarassmentDetector
        print(json.dumps(HarassmentDetector.load(args.model).analyze(args.text).to_dict(), indent=2))


if __name__ == "__main__":
    main()
