# --- Paths
import sys
from pathlib import Path

project_root = Path().resolve().parent
sys.path.append(str(project_root))

# --- Imports
import os
from dotenv import load_dotenv
from huggingface_hub import login
from src.pricer.data_prep.items import Item
from tqdm.notebook import tqdm
from transformers import AutoTokenizer

def fine_tunnning_data_processing(
    lite_mode: bool = False,
    username: str = "ed-donner",
    verbose: bool = True,
    base_model: str = "meta-llama/Llama-3.2-3B",
    token_cutoff: int = 110,
    token_cutoff_details: bool = True,
    push_to_hub: bool = False,
):

    # --- Auth
    load_dotenv(override=True)
    hf_token = os.environ["HF_TOKEN"]
    login(hf_token, add_to_git_credential=True)

    # --- Load data
    dataset = f"{username}/items_lite" if lite_mode else f"{username}/items_full"

    train, val, test = Item.from_hub(dataset)
    items = train + val + test
    if verbose:
        print(
            f"Loaded {len(train):,} training items, {len(val):,} validation items, {len(test):,} test items"
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    if token_cutoff_details:
        token_counts = [item.count_tokens(tokenizer) for item in tqdm(items)]
        cut = len([count for count in token_counts if count > token_cutoff])
        print(
            f"With this CUTOFF, we will truncate {cut:,} items which is {cut/len(items):.1%}"
        )

    # --- Truncate tokens
    for item in tqdm(train + val):
        item.make_prompts(tokenizer, token_cutoff, True)
    for item in tqdm(test):
        item.make_prompts(tokenizer, token_cutoff, False)

    # --- Push to hub

    if push_to_hub:
        dataset = (
            f"{username}/items_prompts_lite"
            if lite_mode
            else f"{username}/items_prompts_full"
        )
        Item.push_prompts_to_hub(dataset, train, val, test)


if __name__ == "__main__":
    fine_tunnning_data_processing()
