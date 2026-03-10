"""
Data preprocessing pipeline: summarise items using LLM batch API.

HOW IT WORKS
────────────
Raw items (loaded from HuggingFace Hub) have no `summary` field yet.
We use the Groq batch API to ask an LLM to generate a concise structured
summary for every item.  Because Groq processes batches asynchronously
(up to 24 h), the pipeline is split into two stages:

  1. submit  – Load raw items, assign IDs, write .jsonl files, upload them
               to Groq, submit as batch jobs, then save the batch state to
               disk (batches.pkl) so the process can safely exit.

  2. poll    – Reload state from disk, repeatedly call Batch.fetch() until
               every batch is complete, verify there are no gaps, then push
               the summarised items to HuggingFace Hub.

  3. run     – Shorthand that runs submit then poll in one process.  Useful
               for the lite dataset which usually finishes in minutes.

USAGE
─────
    # Full dataset – two separate steps (recommended for 820k items)
    python -m src.pricer.data_prep.preprocessing_orchestration --mode submit
    python -m src.pricer.data_prep.preprocessing_orchestration --mode poll --push-to-hub

    # Lite dataset – everything in one go
    python -m src.pricer.data_prep.preprocessing_orchestration --mode run --lite --push-to-hub

    # Custom HuggingFace username
    python -m src.pricer.data_prep.preprocessing_orchestration --mode run --lite \\
        --username YourHFUsername --push-to-hub

    # Adjust how often the poll loop checks Groq (seconds, default 60)
    python -m src.pricer.data_prep.preprocessing_orchestration --mode poll \\
        --poll-interval 120 --push-to-hub
"""

import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.pricer.data_prep.batch import Batch
from src.pricer.data_prep.items import Item

DEFAULT_USERNAME = "Alejandrofupi"

# HF repo names written by the curation pipeline (our input)
RAW_FULL_REPO = "{username}/items_raw_full"
RAW_LITE_REPO = "{username}/items_raw_lite"

# HF repo names we write (our output)
FULL_REPO = "{username}/items_full"
LITE_REPO = "{username}/items_lite"

# Dataset split boundaries – must match data_curation_orchestration.py
TRAIN_FULL = 800_000
VAL_FULL = 810_000
TRAIN_LITE = 20_000
VAL_LITE = 21_000


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline stages
# ──────────────────────────────────────────────────────────────────────────────


def load_items(lite: bool, username: str) -> list[Item]:
    """Load the curated-but-unsummarised items from HuggingFace Hub."""
    repo = (RAW_LITE_REPO if lite else RAW_FULL_REPO).format(username=username)
    print(f"Loading items from {repo} ...")
    train, val, test = Item.from_hub(repo)
    items = train + val + test
    print(f"Loaded {len(items):,} items")
    return items


def assign_ids(items: list[Item]) -> None:
    """Give every item a unique integer ID used as the batch custom_id."""
    for index, item in enumerate(items):
        item.id = index


def submit(lite: bool, username: str) -> list[Item]:
    """
    Stage 1 – Create .jsonl files, upload to Groq, submit batch jobs, save state.

    After this function returns you can safely close the terminal.
    Resume with the `poll` stage.
    """
    items = load_items(lite, username)
    assign_ids(items)

    print(f"\nCreating batches (batch size = {Batch.BATCH_SIZE:,}) ...")
    Batch.create(items, lite)

    print("Submitting batches to Groq ...")
    Batch.run()

    print("Saving batch state to batches.pkl ...")
    Batch.save()

    print("\nSubmit stage complete.")
    print("Run with --mode poll to fetch results once Groq has finished processing.")
    return items


def poll(lite: bool, username: str, push: bool, interval: int) -> list[Item]:
    """
    Stage 2 – Reload state, wait for all batches to finish, push to Hub.

    Calls Batch.fetch() every `interval` seconds until every batch is done,
    then verifies no summaries are missing and optionally pushes to HF Hub.
    """
    # We need items in memory so that Batch.apply_output() can write summaries
    # back into them.  Load fresh from Hub (IDs will be reassigned to match
    # the order used during submit).
    items = load_items(lite, username)
    assign_ids(items)

    print("Loading batch state from batches.pkl ...")
    Batch.load(items)

    print("\nPolling Groq for batch results ...")
    while True:
        Batch.fetch()

        finished = sum(1 for b in Batch.batches if b.done)
        total = len(Batch.batches)
        print(f"  {finished}/{total} batches complete")

        if finished == total:
            break

        print(f"  Waiting {interval}s before next check ...")
        time.sleep(interval)

    # ── Verify no items are missing a summary ──────────────────────────────
    missing = [item.id for item in items if not item.summary]
    if missing:
        print(f"\nWARNING: {len(missing)} items have no summary: {missing[:10]} ...")
    else:
        print(f"\nAll {len(items):,} items have a summary.")

    if push:
        push_to_hub(items, lite, username)
    else:
        print("\nSkipping HuggingFace push (use --push-to-hub to enable).")

    return items


def push_to_hub(items: list[Item], lite: bool, username: str) -> None:
    """Push summarised items to HuggingFace Hub (both full and lite repos)."""
    import os
    from huggingface_hub import login

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("HF_TOKEN environment variable is not set.")
    login(hf_token, add_to_git_credential=True)

    full_repo = FULL_REPO.format(username=username)
    lite_repo = LITE_REPO.format(username=username)

    if lite:
        # Lite dataset: fixed small splits
        train = items[:TRAIN_LITE]
        val = items[TRAIN_LITE:VAL_LITE]
        test = items[VAL_LITE:]
        print(f"\nPushing lite dataset to {lite_repo} ...")
        print(f"  train={len(train):,}  val={len(val):,}  test={len(test):,}")
        Item.push_to_hub(lite_repo, train, val, test)
    else:
        # Full dataset: push full splits, then also push lite subsets
        train = items[:TRAIN_FULL]
        val = items[TRAIN_FULL:VAL_FULL]
        test = items[VAL_FULL:]

        print(f"\nPushing full dataset to {full_repo} ...")
        print(f"  train={len(train):,}  val={len(val):,}  test={len(test):,}")
        Item.push_to_hub(full_repo, train, val, test)

        train_lite = train[:20_000]
        val_lite = val[:1_000]
        test_lite = test[:1_000]

        print(f"Pushing lite dataset to {lite_repo} ...")
        print(f"  train={len(train_lite):,}  val={len(val_lite):,}  test={len(test_lite):,}")
        Item.push_to_hub(lite_repo, train_lite, val_lite, test_lite)

    print("Done pushing to HuggingFace Hub.")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


def run(lite: bool, username: str, push: bool, interval: int) -> None:
    """Convenience: submit then immediately poll (suitable for lite/fast runs)."""
    submit(lite, username)
    poll(lite, username, push=push, interval=interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise Amazon product items using the Groq batch API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["submit", "poll", "run"],
        default="run",
        help="Pipeline stage to execute (default: run)",
    )
    parser.add_argument(
        "--lite",
        action="store_true",
        default=False,
        help="Use the lite (small) dataset instead of the full one",
    )
    parser.add_argument(
        "--username",
        type=str,
        default=DEFAULT_USERNAME,
        help=f"HuggingFace username (default: {DEFAULT_USERNAME})",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        default=False,
        help="Push summarised items to HuggingFace Hub after polling completes",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        dest="poll_interval",
        help="Seconds to wait between Groq status checks during poll (default: 60)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv(override=True)
    args = parse_args()

    if args.mode == "submit":
        submit(lite=args.lite, username=args.username)
    elif args.mode == "poll":
        poll(
            lite=args.lite,
            username=args.username,
            push=args.push_to_hub,
            interval=args.poll_interval,
        )
    else:  # run
        run(
            lite=args.lite,
            username=args.username,
            push=args.push_to_hub,
            interval=args.poll_interval,
        )
