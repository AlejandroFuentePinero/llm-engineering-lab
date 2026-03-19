# NOTE: Local (Apple Silicon) version of basemodel_llama_eval_benchmark.py.
# Runs on MPS (Metal) instead of CUDA — no quantization, model loaded in
# bfloat16. Requires ~6 GB of unified memory for Llama-3.2-3B.
# For the quantized (4-bit) version designed for Google Colab, see
# basemodel_llama_eval_benchmark.py.

# --- Paths
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

# --- Imports

import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from src.pricer.evaluation.basemodel_llama_evaluator import Tester


# --- Benchmark
def basemodel_llama_eval_local(
    verbose: bool = True,
    username: str = "ed-donner",
    lite_mode: bool = True,  # for faster local run
    base_model: str = "meta-llama/Llama-3.2-3B",
):
    """
    Load and evaluate a Llama base model locally on Apple Silicon (MPS).

    Unlike the Colab version, this does not use BitsAndBytesConfig quantization
    (CUDA-only). Instead the model is loaded in bfloat16 and placed on the MPS
    device, falling back to CPU if MPS is unavailable. Expect ~6 GB of unified
    memory usage for Llama-3.2-3B.

    Args:
        verbose:    Print memory footprint and dataset sizes when True.
        username:   Hugging Face username that hosts the items-prompts dataset.
        lite_mode:  Use the smaller *_lite dataset variant when True.
        base_model: Hugging Face model ID to load (e.g. "meta-llama/Llama-3.2-3B").

    Returns:
        dict with keys:
            "model"       – the Hugging Face model ID string used for evaluation.
            "predictions" – list of raw string predictions from the model.
            "test_data"   – the test split of the dataset.
            "tester"      – the Tester instance (contains metrics and guesses).
    """
    # --- Auth
    load_dotenv(override=True)
    hf_token = os.environ["HF_TOKEN"]
    login(hf_token, add_to_git_credential=True)

    # --- Device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if verbose:
        print(f"Using device: {device}")

    # --- Load data
    dataset_name = (
        f"{username}/items_prompts_lite"
        if lite_mode
        else f"{username}/items_prompts_full"
    )

    dataset = load_dataset(dataset_name)
    train = dataset["train"]
    val = dataset["val"]
    test = dataset["test"]

    if verbose:
        print(
            f"Loaded {len(train):,} training items, {len(val):,} validation items, {len(test):,} test items"
        )

    # --- Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    llm = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
    ).to(device)
    llm.generation_config.pad_token_id = tokenizer.pad_token_id

    if verbose:
        print(f"Memory footprint: {llm.get_memory_footprint() / 1e9:.1f} GB")

    # --- Model predictor
    def model_predict(item):
        inputs = tokenizer(item["prompt"], return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = llm.generate(**inputs, max_new_tokens=8)
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, prompt_len:]
        return tokenizer.decode(generated_ids)

    # --- Evaluate LLM performance (workers=1 to avoid shared HTTP client issues across threads)
    tester = Tester(model_predict, test)
    tester.run()

    return {
        "model": base_model,
        "predictions": tester.guesses,
        "test_data": test,
        "tester": tester,
    }


if __name__ == "__main__":
    basemodel_llama_eval_local()
