# --- Paths
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

# --- Imports
import os
import chromadb
from sentence_transformers import SentenceTransformer
from litellm import completion
from dotenv import load_dotenv
from huggingface_hub import login
from src.pricer.agents.evaluator import Tester
from src.pricer.agents.items import Item


def rag_pipeline(
    collection_name: str = "products",
    db: str = "products_vectorstore",
    lite_mode: bool = False,
    username: str = "ed-donner",
    verbose: bool = True,
):
    """
    Benchmark GPT-5.1 with Retrieval-Augmented Generation for price prediction.

    For each test item, retrieves the 5 most similar products from the ChromaDB
    vectorstore (built by rag_ingest.py) using cosine similarity on MiniLM-L6-v2
    embeddings. Those similar products and their prices are injected as context
    into the prompt before calling GPT-5.1, grounding the prediction in real
    comparable products rather than relying on parametric knowledge alone.

    Evaluates on 200 test items and reports MAE, MSE, and R² via the Tester class.
    Requires the vectorstore to be built first (run rag_ingest.py).
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

    # --- Load ChromaDB
    db_path = str(project_root / db)
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(collection_name)

    # --- Sentence transformer
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # --- Find similar products
    def find_similars(description: str):
        vec = encoder.encode([description]).astype(float).tolist()
        results = collection.query(query_embeddings=vec, n_results=5)
        documents = results["documents"][0]
        prices = [m["price"] for m in results["metadatas"][0]]
        return documents, prices

    # --- Build context from similar products
    def make_context(similars, prices):
        message = "For context, here are some other items that might be similar to the item you need to estimate.\n\n"
        for similar, price in zip(similars, prices):
            message += (
                f"Potentially related product:\n{similar}\nPrice is ${price:.2f}\n\n"
            )
        return message

    # --- Build messages for the LLM
    def messages_for(item):
        similars, prices = find_similars(item.summary)
        message = f"Estimate the price of this product. Respond with the price, no explanation\n\n{item.summary}\n\n"
        message += make_context(similars, prices)
        return [{"role": "user", "content": message}]

    # --- RAG predictor
    def gpt_5_1_rag(item):
        response = completion(
            model="gpt-5.1",
            messages=messages_for(item),
            reasoning_effort="none",
            seed=42,
        )
        return response.choices[0].message.content

    # --- Evaluate
    tester = Tester(gpt_5_1_rag, test, workers=1)
    tester.run()

    return {
        "model": "gpt-5.1",
        "predictions": tester.guesses,
        "test_data": test,
        "tester": tester,
    }
