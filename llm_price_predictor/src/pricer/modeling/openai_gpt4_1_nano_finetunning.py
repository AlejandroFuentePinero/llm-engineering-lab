# --- Path
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

# --- Imports

import os
import json
import time
from dotenv import load_dotenv
from huggingface_hub import login
from openai import OpenAI
from src.pricer.data_prep.items import Item
from src.pricer.evaluation.evaluator import evaluate


# --- Auth helper (called once, shared across functions)

def _setup_auth() -> tuple[OpenAI, list, list, list]:
    """Load env, authenticate HF, and return (openai_client, train, val, test)."""
    raise RuntimeError("Use _load_auth() and Item.from_hub() separately; this is a placeholder.")


def _load_env():
    load_dotenv(override=True)


def _hf_login():
    hf_token = os.environ["HF_TOKEN"]
    login(hf_token, add_to_git_credential=True)


# --- Prompt builders

def messages_for(item):
    message = f"Estimate the price of this product. Respond with the price, no explanation\n\n{item.summary}"
    return [
        {"role": "user", "content": message},
        {"role": "assistant", "content": f"${item.price:.2f}"},
    ]


def test_messages_for(item):
    message = f"Estimate the price of this product. Respond with the price, no explanation\n\n{item.summary}"
    return [{"role": "user", "content": message}]


# --- JSONL helpers

def make_jsonl(items):
    result = ""
    for item in items:
        result += json.dumps({"messages": messages_for(item)}) + "\n"
    return result.strip()


def write_jsonl(fine_tune_train, fine_tune_validation, jsonl_dir: Path):
    with open(jsonl_dir / "fine_tune_train.jsonl", "w") as f:
        f.write(make_jsonl(fine_tune_train))
    with open(jsonl_dir / "fine_tune_validation.jsonl", "w") as f:
        f.write(make_jsonl(fine_tune_validation))
    print("Done writing JSONL files")


# --- Fine-tuning

def openai_finetuning(
    verbose: bool = True,
    username: str = "ed-donner",
    lite_mode: bool = False,
    model: str = "gpt-4.1-nano-2025-04-14",
    train_size: int = 20_000,
    validation_size: int = 50,
    jsonl_dir: Path = project_root / "jsonl",
):
    _load_env()
    _hf_login()

    openai = OpenAI()

    dataset = f"{username}/items_lite" if lite_mode else f"{username}/items_full"
    train, val, test = Item.from_hub(dataset)

    if verbose:
        print(
            f"Loaded {len(train):,} training items, {len(val):,} validation items, {len(test):,} test items"
        )

    fine_tune_train = train[:train_size]
    fine_tune_validation = val[:validation_size]

    write_jsonl(fine_tune_train, fine_tune_validation, jsonl_dir)

    with open(jsonl_dir / "fine_tune_train.jsonl", "rb") as f:
        train_file = openai.files.create(
            file=("fine_tune_train.jsonl", f), purpose="fine-tune"
        )

    with open(jsonl_dir / "fine_tune_validation.jsonl", "rb") as f:
        validation_file = openai.files.create(
            file=("fine_tune_validation.jsonl", f), purpose="fine-tune"
        )

    job = openai.fine_tuning.jobs.create(
        training_file=train_file.id,
        validation_file=validation_file.id,
        model=model,
        seed=42,
        hyperparameters={"n_epochs": 1, "batch_size": 16},
        suffix="pricer",
    )

    print(f"Fine-tuning job submitted: {job.id}")
    return job.id


# --- Status & model name

def finetuning_status(job_id: str | None = None) -> list:
    _load_env()
    openai = OpenAI()
    if job_id is None:
        job_id = openai.fine_tuning.jobs.list(limit=1).data[0].id
    return openai.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=10).data


def finetuned_model_name(job_id: str | None = None) -> str | None:
    _load_env()
    openai = OpenAI()
    if job_id is None:
        job_id = openai.fine_tuning.jobs.list(limit=1).data[0].id
    return openai.fine_tuning.jobs.retrieve(job_id).fine_tuned_model


# --- Wait for job completion (orchestration helper)

def wait_for_finetuning(job_id: str | None = None, poll_interval: int = 60, verbose: bool = True) -> str:
    """
    Poll OpenAI until the fine-tuning job succeeds or fails.
    Returns the fine-tuned model name on success, raises on failure.
    """
    _load_env()
    openai = OpenAI()
    if job_id is None:
        job_id = openai.fine_tuning.jobs.list(limit=1).data[0].id

    terminal_states = {"succeeded", "failed", "cancelled"}
    if verbose:
        print(f"Waiting for fine-tuning job {job_id} (polling every {poll_interval}s) ...")

    while True:
        job = openai.fine_tuning.jobs.retrieve(job_id)
        status = job.status

        if verbose:
            print(f"  [{time.strftime('%H:%M:%S')}] status={status}")

        if status in terminal_states:
            if status != "succeeded":
                raise RuntimeError(f"Fine-tuning job {job_id} ended with status: {status}")
            model_name = job.fine_tuned_model
            if verbose:
                print(f"Fine-tuned model ready: {model_name}")
            return model_name

        time.sleep(poll_interval)


# --- Prediction using fine-tuned model

def make_fine_tuned_predictor(model_name: str):
    """
    Returns a callable that predicts price for a single item.
    Resolves the model name once, avoiding a round-trip per prediction.
    """
    openai = OpenAI()

    def predict(item):
        response = openai.chat.completions.create(
            model=model_name,
            messages=test_messages_for(item),
            seed=42,
            max_tokens=7,
        )
        return response.choices[0].message.content

    predict.__name__ = f"fine_tuned_openai_{model_name.split(':')[-1]}"
    return predict


# --- Evaluation

def fine_tune_evaluation(
    model_name: str | None = None,
    verbose: bool = True,
    username: str = "ed-donner",
    lite_mode: bool = False,
):
    _load_env()
    _hf_login()

    if model_name is None:
        model_name = finetuned_model_name()

    dataset = f"{username}/items_lite" if lite_mode else f"{username}/items_full"
    train, val, test = Item.from_hub(dataset)

    if verbose:
        print(f"Evaluating model: {model_name}")

    predictor = make_fine_tuned_predictor(model_name)
    evaluate(predictor, test)


# --- Full orchestration (submit → wait → evaluate)

def run_finetuning_pipeline(
    verbose: bool = True,
    username: str = "ed-donner",
    lite_mode: bool = False,
    model: str = "gpt-4.1-nano-2025-04-14",
    train_size: int = 20_000,
    validation_size: int = 50,
    jsonl_dir: Path = project_root / "jsonl",
    poll_interval: int = 60,
):
    """
    End-to-end pipeline:
      1. Submit fine-tuning job
      2. Poll until complete
      3. Evaluate the resulting model on the test set
    """
    job_id = openai_finetuning(
        verbose=verbose,
        username=username,
        lite_mode=lite_mode,
        model=model,
        train_size=train_size,
        validation_size=validation_size,
        jsonl_dir=jsonl_dir,
    )

    model_name = wait_for_finetuning(job_id=job_id, poll_interval=poll_interval, verbose=verbose)

    fine_tune_evaluation(
        model_name=model_name,
        verbose=verbose,
        username=username,
        lite_mode=lite_mode,
    )

    return model_name
