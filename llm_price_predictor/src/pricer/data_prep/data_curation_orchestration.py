"""
Data curation pipeline for the price predictor.

Loads Amazon product datasets, deduplicates, resamples by price distribution,
and optionally pushes the curated splits to HuggingFace Hub.

Usage:
    python -m src.pricer.data_prep.data_curation_orchestration [options]

Examples:
    # Run pipeline locally (no HF push)
    python -m src.pricer.data_prep.data_curation_orchestration

    # Run and push to HuggingFace
    python -m src.pricer.data_prep.data_curation_orchestration --push-to-hub --username Alejandrofupi

    # Custom sample size
    python -m src.pricer.data_prep.data_curation_orchestration --size 400000 --push-to-hub
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.pricer.data_prep.items import Item
from src.pricer.data_prep.loaders import (
    ItemLoader,
)

DATASET_NAMES = [
    "Automotive",
    "Electronics",
    "Office_Products",
    "Tools_and_Home_Improvement",
    "Cell_Phones_and_Accessories",
    "Toys_and_Games",
    "Appliances",
    "Musical_Instruments",
]

DEFAULT_SIZE = 820_000
DEFAULT_USERNAME = "Alejandrofupi"

TRAIN_SPLIT = 800_000
VAL_SPLIT = 810_000

LITE_TRAIN = 20_000
LITE_VAL = 1_000
LITE_TEST = 1_000


def load_all(dataset_names: list[str]) -> list[Item]:
    items = []
    for name in dataset_names:
        loader = ItemLoader(name)
        items.extend(loader.load())
    print(f"\nGrand total: {len(items):,} items loaded")
    return items


def deduplicate(items: list[Item]) -> list[Item]:
    random.seed(42)
    random.shuffle(items)

    seen: set = set()
    items = [
        x
        for x in tqdm(items, desc="Dedup by title")
        if not (x.title in seen or seen.add(x.title))
    ]

    seen = set()
    items = [
        x
        for x in tqdm(items, desc="Dedup by full text")
        if not (x.full in seen or seen.add(x.full))
    ]

    print(f"After deduplication: {len(items):,} items")
    return items


def resample(items: list[Item], size: int) -> list[Item]:
    np.random.seed(42)

    prices = np.array([it.price for it in items], dtype=float)
    categories = np.array([it.category for it in items])

    p = (prices - prices.min()) / (prices.max() - prices.min() + 1e-9)
    w = p**2
    w[categories == "Tools_and_Home_Improvement"] *= 0.5
    w[categories == "Automotive"] *= 0.05

    # Ensure every item has a non-zero weight so weighted sampling without
    # replacement never runs out of eligible candidates.
    w = np.clip(w, a_min=1e-12, a_max=None)
    w = w / w.sum()

    idx = np.random.choice(len(items), size=size, replace=False, p=w)
    sample = [items[i] for i in idx]
    print(f"Resampled to {len(sample):,} items")
    return sample


def push_to_hub(sample: list[Item], username: str):
    from huggingface_hub import login

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("HF_TOKEN environment variable is not set.")
    login(hf_token, add_to_git_credential=True)

    full_repo = f"{username}/items_raw_full"
    lite_repo = f"{username}/items_raw_lite"

    train = sample[:TRAIN_SPLIT]
    val = sample[TRAIN_SPLIT:VAL_SPLIT]
    test = sample[VAL_SPLIT:]

    print(f"\nPushing full dataset to {full_repo} ...")
    print(f"  train={len(train):,}  val={len(val):,}  test={len(test):,}")
    Item.push_to_hub(full_repo, train, val, test)

    train_lite = train[:LITE_TRAIN]
    val_lite = val[:LITE_VAL]
    test_lite = test[:LITE_TEST]

    print(f"Pushing lite dataset to {lite_repo} ...")
    print(
        f"  train={len(train_lite):,}  val={len(val_lite):,}  test={len(test_lite):,}"
    )
    Item.push_to_hub(lite_repo, train_lite, val_lite, test_lite)

    print("Done pushing to HuggingFace Hub.")


def run(size: int = DEFAULT_SIZE, push: bool = False, username: str = DEFAULT_USERNAME):
    load_dotenv(override=True)

    items = load_all(DATASET_NAMES)
    items = deduplicate(items)
    sample = resample(items, size=size)

    if push:
        push_to_hub(sample, username=username)
    else:
        print("\nSkipping HuggingFace push (use --push-to-hub to enable).")

    return sample


def parse_args():
    parser = argparse.ArgumentParser(
        description="Curate Amazon product data for the price predictor."
    )
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SIZE,
        help=f"Number of items to keep after resampling (default: {DEFAULT_SIZE:,})",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        default=False,
        help="Push curated datasets to HuggingFace Hub",
    )
    parser.add_argument(
        "--username",
        type=str,
        default=DEFAULT_USERNAME,
        help=f"HuggingFace username for push (default: {DEFAULT_USERNAME})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(size=args.size, push=args.push_to_hub, username=args.username)
