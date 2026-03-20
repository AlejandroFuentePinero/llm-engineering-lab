# NOTE: This script requires a CUDA-capable GPU and is intended to be run in
# Google Colab (Runtime → Change runtime type → T4 GPU). It will not run
# locally on a Mac due to the BitsAndBytesConfig quantization dependency.

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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from datasets import load_dataset

from src.pricer.evaluation.basemodel_llama_evaluator import Tester


# --- Benchmark
def basemodel_llama_eval(
    verbose: bool = True,
    username: str = "ed-donner",
    lite_mode: bool = True,  # for faster local run
    base_model: str = "meta-llama/Llama-3.2-3B",
    quant_4_bit: bool = True,
):
    """
    Load and evaluate a Llama base model (no fine-tuning) on the price prediction task.

    The model is loaded with 4-bit (NF4) or 8-bit quantization to reduce GPU memory
    usage, then run on the test split of the items-prompts dataset. The raw text
    output of each generation is passed to Tester, which extracts the numeric price
    and computes standard regression metrics (MAE, RMSE, etc.).

    Args:
        verbose:     Print memory footprint and dataset sizes when True.
        username:    Hugging Face username that hosts the items-prompts dataset.
        lite_mode:   Use the smaller *_lite dataset variant when True.
        base_model:  Hugging Face model ID to load (e.g. "meta-llama/Llama-3.2-3B").
        quant_4_bit: Use 4-bit NF4 quantization when True, 8-bit otherwise.

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

    # --- Prepare quantization config
    if quant_4_bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True, bnb_8bit_compute_dtype=torch.bfloat16
        )

    # --- Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    llm = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map="auto",
    )
    llm.generation_config.pad_token_id = tokenizer.pad_token_id

    if verbose:
        print(f"Memory footprint: {llm.get_memory_footprint() / 1e9:.1f} GB")

    # --- Model predictor
    def model_predict(item):
        inputs = tokenizer(item["prompt"], return_tensors="pt").to("cuda")
        with torch.no_grad():
            output_ids = llm.generate(**inputs, max_new_tokens=8)
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, prompt_len:]
        return tokenizer.decode(generated_ids)

    # --- Evaluate LLM performance
    tester = Tester(model_predict, test)
    tester.run()

    return {
        "model": base_model,
        "predictions": tester.guesses,
        "test_data": test,
        "tester": tester,
    }
