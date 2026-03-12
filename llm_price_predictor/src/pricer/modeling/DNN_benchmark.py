# --- Paths
import sys
from pathlib import Path

project_root = Path().resolve().parent
sys.path.append(str(project_root))

# --- Imports
import os
from dotenv import load_dotenv
from huggingface_hub import login
from src.pricer.evaluation.evaluator import Tester
from src.pricer.data_prep.items import Item
from src.pricer.modeling.deep_neural_network import DeepNeuralNetworkRunner


# --- Benchmark
def DNN_benchmark(
    verbose: bool = True,
    epochs: int = 5,
    val_size: int = 1000,
    username: str = "ed-donner",
    lite_mode: bool = False,
    save_path: str = None,
    evaluation: bool = True,
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

    # --- Setup runner
    runner = DeepNeuralNetworkRunner(train, val[:val_size])
    runner.setup()

    # --- Train
    runner.train(epochs=epochs)

    # --- Optionally save the model
    if save_path:
        runner.save(save_path)
        if verbose:
            print(f"Model saved to {save_path}")

    # --- Predictor
    def deep_neural_network(item):
        return runner.inference(item)

    # --- Evaluate
    tester = None
    if evaluation:
        tester = Tester(deep_neural_network, test)
        tester.run()

    predictions = [deep_neural_network(item) for item in test]

    return {
        "runner": runner,
        "predictor": deep_neural_network,
        "predictions": predictions,
        "test_data": test,
        "tester": tester,
    }


if __name__ == "__main__":
    DNN_benchmark()
