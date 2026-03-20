# NOTE: This script requires a CUDA-capable GPU and is intended to be run in
# Google Colab (Runtime → Change runtime type → T4 GPU). It will not run
# locally.

# --- Paths
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

# --- Imports
import os
import re
import math
from tqdm import tqdm
from google.colab import userdata
from huggingface_hub import login
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed, BitsAndBytesConfig
from datasets import load_dataset, Dataset, DatasetDict
import wandb
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datetime import datetime
import matplotlib.pyplot as plt


def llama_ft_training(LOG_TO_WANDB: bool = True,
                      QUANT_4_BIT: bool = True,
                      HF_USER: str = "Alejandrofupi",
                      BASE_MODEL: str = "meta-llama/Llama-3.2-3B",
                      LITE_MODE: bool = True,
                      PROJECT_NAME: str = "price",
                      DATA_USER: str = "ed-donner",
                      OPTIMIZER: str = "paged_adamw_32bit",
                      PUSH_TO_HUB: bool = False):
    """
    Fine-tune Llama-3.2-3B on the price-prediction dataset using QLoRA.

    Loads the items-prompts dataset from HuggingFace Hub, applies quantised LoRA
    fine-tuning via SFTTrainer, and optionally logs training metrics to Weights &
    Biases and pushes checkpoints to HuggingFace Hub.

    Intended to run on a CUDA GPU in Google Colab (T4 or better). Secrets
    (HF_TOKEN, WANDB_API_KEY) are read from Colab's userdata store
    (Tools → Secrets).

    Args:
        LOG_TO_WANDB:  Log training metrics to Weights & Biases when True.
        QUANT_4_BIT:   Use 4-bit NF4 quantisation when True, 8-bit otherwise.
        HF_USER:       HuggingFace username to push checkpoints to.
        BASE_MODEL:    HuggingFace model ID to fine-tune.
        LITE_MODE:     Use the smaller *_lite dataset and lighter hyperparameters.
        PROJECT_NAME:  W&B project name and output directory prefix.
        DATA_USER:     HuggingFace username that hosts the items-prompts dataset.
        OPTIMIZER:     Optimiser name passed to SFTConfig.
        PUSH_TO_HUB:   Push the final model to HuggingFace Hub when True.
    """

    # --- Dataset name
    DATASET_NAME = f"{DATA_USER}/items_prompts_lite" if LITE_MODE else f"{DATA_USER}/items_prompts_full"

    RUN_NAME = f"{datetime.now():%Y-%m-%d_%H.%M.%S}"
    if LITE_MODE:
        RUN_NAME += "-lite"
    PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
    HUB_MODEL_NAME = f"{HF_USER}/{PROJECT_RUN_NAME}"

    # --- Hyper-parameters: overall
    EPOCHS = 1 if LITE_MODE else 3
    BATCH_SIZE = 32 if LITE_MODE else 256
    MAX_SEQUENCE_LENGTH = 128
    GRADIENT_ACCUMULATION_STEPS = 1

    # --- Hyper-parameters: QLoRA
    LORA_R = 32 if LITE_MODE else 256
    LORA_ALPHA = LORA_R * 2
    ATTENTION_LAYERS = ["q_proj", "v_proj", "k_proj", "o_proj"]
    MLP_LAYERS = ["gate_proj", "up_proj", "down_proj"]
    TARGET_MODULES = ATTENTION_LAYERS if LITE_MODE else ATTENTION_LAYERS + MLP_LAYERS
    LORA_DROPOUT = 0.1

    # --- Hyper-parameters: training
    LEARNING_RATE = 1e-4
    WARMUP_RATIO = 0.01
    LR_SCHEDULER_TYPE = "cosine"
    WEIGHT_DECAY = 0.001

    capability = torch.cuda.get_device_capability()
    use_bf16 = capability[0] >= 8

    # --- Tracking
    VAL_SIZE = 500 if LITE_MODE else 1000
    LOG_STEPS = 5 if LITE_MODE else 10
    SAVE_STEPS = 100 if LITE_MODE else 200

    # --- Auth: HuggingFace
    hf_token = userdata.get("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN not found in Colab secrets. "
            "Add it via Tools → Secrets and enable notebook access."
        )
    login(hf_token, add_to_git_credential=True)

    # --- Auth: Weights & Biases
    if LOG_TO_WANDB:
        wandb_api_key = userdata.get("WANDB_API_KEY")
        if not wandb_api_key:
            raise ValueError(
                "WANDB_API_KEY not found in Colab secrets. "
                "Add it via Tools → Secrets or set LOG_TO_WANDB=False."
            )
        os.environ["WANDB_API_KEY"] = wandb_api_key
        os.environ["WANDB_PROJECT"] = PROJECT_NAME
        os.environ["WANDB_LOG_MODEL"] = "false"
        os.environ["WANDB_WATCH"] = "false"
        wandb.login()
        wandb.init(project=PROJECT_NAME, name=RUN_NAME)

    # --- Load dataset
    dataset = load_dataset(DATASET_NAME)
    train = dataset["train"]
    val = dataset["val"]

    # --- Quantisation config
    if QUANT_4_BIT:
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

    # --- Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto",
    )
    base_model.generation_config.pad_token_id = tokenizer.pad_token_id

    # --- LoRA parameters
    lora_parameters = LoraConfig(
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        r=LORA_R,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )

    # --- Training parameters
    train_parameters = SFTConfig(
        output_dir=PROJECT_RUN_NAME,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        optim=OPTIMIZER,
        save_steps=SAVE_STEPS,
        save_total_limit=10,
        logging_steps=LOG_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fp16=not use_bf16,
        bf16=use_bf16,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=WARMUP_RATIO,
        group_by_length=True,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        report_to="wandb" if LOG_TO_WANDB else None,
        run_name=RUN_NAME,
        max_length=MAX_SEQUENCE_LENGTH,
        save_strategy="steps",
        hub_strategy="every_save",
        push_to_hub=PUSH_TO_HUB,
        hub_model_id=HUB_MODEL_NAME,
        hub_private_repo=True,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
    )

    # --- Trainer
    fine_tuning = SFTTrainer(
        model=base_model,
        train_dataset=train,
        eval_dataset=val,
        peft_config=lora_parameters,
        args=train_parameters,
    )

    # --- Fine-tuning
    fine_tuning.train()

    if PUSH_TO_HUB:
        fine_tuning.model.push_to_hub(PROJECT_RUN_NAME, private=True)
        print(f"Saved to the hub: {PROJECT_RUN_NAME}")

    if LOG_TO_WANDB:
        wandb.finish()
