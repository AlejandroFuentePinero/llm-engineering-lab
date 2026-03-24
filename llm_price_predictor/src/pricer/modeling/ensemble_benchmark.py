# --- Paths
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

# --- Imports
import os
import re
import chromadb
from dotenv import load_dotenv
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from litellm import completion
from src.pricer.agents.items import Item
from src.pricer.agents.evaluator import Tester
from src.pricer.agents.deep_neural_network import DeepNeuralNetworkInference
from src.pricer.agents.specialist_agent import SpecialistAgent

## Download the Neural Network weights from Week 6 into the modeling directory:
## https://drive.google.com/drive/folders/1uq5C9edPIZ1973dArZiEO-VE13F7m8MK?usp=drive_link
DNN_PATH = str(project_root / "src/pricer/modeling/deep_neural_network.pth")


def ensemble_benchmark(
    lite_mode: bool = False,
    username: str = "ed-donner",
    db: str = "products_vectorstore",
    collection_name: str = "products",
    verbose: bool = True,
):
    """
    Benchmark an ensemble model combining three complementary price predictors.

    The ensemble weights predictions from:
      - GPT-5.1 + RAG (80%): frontier LLM grounded by 5 retrieved similar products
      - Fine-tuned specialist on Modal (10%): domain-adapted model deployed remotely
      - Deep Neural Network (10%): fast local predictor using hashed text features

    The RAG model dominates the blend; the specialist and DNN act as anchors
    that dampen implausible outliers from the LLM.

    Prerequisites:
      - ChromaDB vectorstore built via rag_ingest.py
      - DNN weights (deep_neural_network.pth) downloaded into src/pricer/modeling/
        from https://drive.google.com/drive/folders/1uq5C9edPIZ1973dArZiEO-VE13F7m8MK
      - Fine-tuned specialist deployed on Modal (pricer-service)
    """
    # --- Auth
    load_dotenv(override=True)
    hf_token = os.environ["HF_TOKEN"]
    login(token=hf_token, add_to_git_credential=False)

    # --- Load data
    dataset = f"{username}/items_lite" if lite_mode else f"{username}/items_full"
    train, val, test = Item.from_hub(dataset)

    if verbose:
        print(
            f"Loaded {len(train):,} training items, {len(val):,} validation items, {len(test):,} test items"
        )

    # --- RAG setup
    db_path = str(project_root / db)
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(collection_name)
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def find_similars(description: str):
        vec = encoder.encode([description]).astype(float).tolist()
        results = collection.query(query_embeddings=vec, n_results=5)
        documents = results["documents"][0]
        prices = [m["price"] for m in results["metadatas"][0]]
        return documents, prices

    def make_context(similars, prices):
        message = "For context, here are some other items that might be similar to the item you need to estimate.\n\n"
        for similar, price in zip(similars, prices):
            message += f"Potentially related product:\n{similar}\nPrice is ${price:.2f}\n\n"
        return message

    def messages_for(item):
        similars, prices = find_similars(item.summary)
        message = f"Estimate the price of this product. Respond with the price, no explanation\n\n{item.summary}\n\n"
        message += make_context(similars, prices)
        return [{"role": "user", "content": message}]

    def gpt_5_1_rag(item):
        response = completion(
            model="gpt-5.1",
            messages=messages_for(item),
            reasoning_effort="none",
            seed=42,
        )
        return response.choices[0].message.content

    # --- DNN setup (same 10-layer residual network trained in DNN_benchmark.py)
    runner = DeepNeuralNetworkInference()
    runner.setup()
    runner.load(DNN_PATH)

    def deep_neural_network(item):
        return runner.inference(item.summary)

    # --- Specialist (fine-tuned model on Modal)
    specialist_agent = SpecialistAgent()

    def specialist(item):
        return specialist_agent.price(item.summary)

    # --- Parse LLM string price to float for weighting
    def get_price(reply):
        reply = str(reply).replace("$", "").replace(",", "")
        match = re.search(r"[-+]?\d*\.\d+|\d+", reply)
        return float(match.group()) if match else 0

    # --- Ensemble: RAG frontier model + fine-tuned specialist + DNN
    def ensemble(item):
        price_rag = get_price(gpt_5_1_rag(item))
        price_specialist = specialist(item)
        price_dnn = deep_neural_network(item)
        return price_rag * 0.8 + price_specialist * 0.1 + price_dnn * 0.1

    # --- Evaluate
    tester = Tester(ensemble, test, workers=1)
    tester.run()

    return {
        "predictions": tester.guesses,
        "test_data": test,
        "tester": tester,
    }


if __name__ == "__main__":
    ensemble_benchmark()
