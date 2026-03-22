# NOTE: This script requires a CUDA-capable GPU and is intended to be run in
# Google Colab (Runtime → Change runtime type → T4 GPU). It will not run
# locally on a Mac due to the BitsAndBytesConfig quantization dependency.

# --- Paths
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

# --- Imports

import torch
from google.colab import userdata
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from datasets import load_dataset
from peft import PeftModel

from src.pricer.evaluation.basemodel_llama_evaluator import Tester


# --- Benchmark
def finetuned_llama_eval(
    verbose: bool = True,
    hf_user: str = "ed-donner",
    data_user: str = "ed-donner",
    lite_mode: bool = False,
    base_model: str = "meta-llama/Llama-3.2-3B",
    project_name: str = "price",
    run_name: str = "2025-11-28_18.47.07",
    revision: str = "b19c8bfea3b6ff62237fbb0a8da9779fc12cefbd",
    quant_4_bit: bool = True,
):
    """
    Load and evaluate a fine-tuned Llama model (via PEFT/LoRA) on the price prediction task.

    The base model is loaded with 4-bit (NF4) or 8-bit quantization, then the
    fine-tuned LoRA adapter is applied via PeftModel. The combined model is run on
    the test split of the items-prompts dataset. Raw text output is passed to Tester,
    which extracts the numeric price and computes standard regression metrics.

    Args:
        verbose:      Print memory footprint and dataset sizes when True.
        hf_user:      Hugging Face username that hosts the fine-tuned adapter.
        data_user:    Hugging Face username that hosts the items-prompts dataset.
        lite_mode:    Use the smaller *_lite dataset variant when True.
        base_model:   Hugging Face model ID to load (e.g. "meta-llama/Llama-3.2-3B").
        project_name: Project prefix used to build the adapter repo name.
        run_name:     Run timestamp used to build the adapter repo name.
        revision:     Specific commit hash of the adapter to load (None = latest).
        quant_4_bit:  Use 4-bit NF4 quantization when True, 8-bit otherwise.

    Returns:
        dict with keys:
            "model"       – the Hugging Face base model ID string used for evaluation.
            "predictions" – list of raw string predictions from the fine-tuned model.
            "test_data"   – the test split of the dataset.
            "tester"      – the Tester instance (contains metrics and guesses).
    """
    # --- Auth
    hf_token = userdata.get("HF_TOKEN")
    login(hf_token, add_to_git_credential=True)

    # --- Load data
    dataset_name = (
        f"{data_user}/items_prompts_lite"
        if lite_mode
        else f"{data_user}/items_prompts_full"
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
    capability = torch.cuda.get_device_capability()
    use_bf16 = capability[0] >= 8

    if quant_4_bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        )

    # --- Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    llm = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map="auto",
    )
    llm.generation_config.pad_token_id = tokenizer.pad_token_id

    # --- Apply fine-tuned LoRA adapter
    hub_model_name = f"{hf_user}/{project_name}-{run_name}"

    if revision:
        fine_tuned_model = PeftModel.from_pretrained(
            llm, hub_model_name, revision=revision
        )
    else:
        fine_tuned_model = PeftModel.from_pretrained(llm, hub_model_name)

    if verbose:
        print(f"Memory footprint: {fine_tuned_model.get_memory_footprint() / 1e9:.1f} GB")

    # --- Model predictor
    def model_predict(item):
        inputs = tokenizer(item["prompt"], return_tensors="pt").to("cuda")
        with torch.no_grad():
            output_ids = fine_tuned_model.generate(**inputs, max_new_tokens=8)
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
