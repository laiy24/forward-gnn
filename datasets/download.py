#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import settings
from datasets.dataloader import load_dataset


EXPERIMENT_DATASETS = [
    "CitationFull-CiteSeer",
    "CitationFull-Cora_ML",
    "CitationFull-PubMed",
    "Amazon-Photo",
    "GitHub",
]


def warmup_dataset(dataset_name: str) -> None:
    print(f"\nDownloading/preparing dataset: {dataset_name}")
    _dataset = load_dataset(dataset_name)
    data = _dataset[0]
    print(
        f"  - ok: nodes={data.num_nodes}, edges={data.num_edges}, "
        f"features={getattr(data, 'num_features', 'n/a')}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download datasets for forwardgnn experiments")
    p.add_argument(
        "--datasets",
        nargs="*",
        default=EXPERIMENT_DATASETS,
        help="Dataset names to prepare. Defaults to all datasets used in exp scripts.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("Downloading all required experiment datasets...")
    print(f"DATA_ROOT: {Path(settings.DATA_ROOT).resolve()}")

    for dataset_name in args.datasets:
        warmup_dataset(dataset_name)

    print("\nDone. All required datasets are ready.")


if __name__ == "__main__":
    main()
