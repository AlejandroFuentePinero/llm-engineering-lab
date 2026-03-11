# --- Paths
import sys
from pathlib import Path

project_root = Path().resolve().parent
sys.path.append(str(project_root))

# --- Imports

import os
from dotenv import load_dotenv
from huggingface_hub import login
from litellm import completion
from src.pricer.evaluation.evaluator import Tester
from src.pricer.data_prep.items import Item


# --- Message generation
def messages_for(item):
    message = f"Estimate the price of this product. Respond with the price, no explanation\n\n{item.summary}"
    return [{"role": "user", "content": message}]


# --- Benchmark
def LLM_benchmark(
    verbose: bool = True,
    username: str = "ed-donner",
    lite_mode: bool = False,
    model: str = "openai/gpt-4.1-nano",
):

    # --- Auth
    load_dotenv(override=True)
    hf_token = os.environ["HF_TOKEN"]
    login(hf_token, add_to_git_credential=True)

    # --- Load data
    dataset = f"{username}/items_lite" if lite_mode else f"{username}/items_full"
    train, val, test = Item.from_hub(dataset)

    if verbose:
        print(
            f"Loaded {len(train):,} training items, {len(val):,} validation items, {len(test):,} test items"
        )

    # --- Call LLM to predict prize
    def LLM_call(item):
        response = completion(model=model, messages=messages_for(item))
        return response.choices[0].message.content

    # --- Evaluate LLM performance (workers=1 to avoid shared HTTP client issues across threads)
    tester = Tester(LLM_call, test, workers=1)
    tester.run()

    return {
        "model": model,
        "predictions": tester.guesses,
        "test_data": test,
        "tester": tester,
    }


if __name__ == "__main__":
    LLM_benchmark()
