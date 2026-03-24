# --- Paths
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

# --- Imports
import os
from dotenv import load_dotenv
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm
from src.pricer.agents.items import Item


def ingest(
    lite_mode: bool = False,
    db: str = "products_vectorstore",
    username: str = "ed-donner",
    verbose: bool = True,
    collection_name: str = "products",
):
    """
    Build a ChromaDB vectorstore from the training dataset.

    Encodes all training product summaries using the sentence-transformers
    all-MiniLM-L6-v2 model and stores the resulting embeddings in a persistent
    ChromaDB collection. Skips ingestion if the collection already exists.

    A test query is run at the end of ingestion to force HNSW index compaction
    before the process exits, ensuring the vectorstore is queryable by subsequent
    processes (e.g. rag_pipeline.py or ensemble_benchmark.py).

    Must be run once before any RAG or ensemble benchmarks.
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

    # --- Chroma datastore (always at project root)
    db_path = str(project_root / db)
    client = chromadb.PersistentClient(path=db_path)

    if verbose:
        print(f"ChromaDB path: {db_path}")

    # --- Sentence transformer
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # --- Populate the database
    existing_collection_names = [
        collection.name for collection in client.list_collections()
    ]

    if collection_name not in existing_collection_names:
        collection = client.create_collection(collection_name)
        for i in tqdm(range(0, len(train), 1000)):
            documents = [item.summary for item in train[i : i + 1000]]
            vectors = encoder.encode(documents).astype(float).tolist()
            metadatas = [
                {"category": item.category, "price": item.price}
                for item in train[i : i + 1000]
            ]
            ids = [f"doc_{j}" for j in range(i, i + 1000)]
            ids = ids[: len(documents)]
            collection.add(
                ids=ids, documents=documents, embeddings=vectors, metadatas=metadatas
            )
        if verbose:
            print(f"Collection '{collection_name}' created with {len(train):,} documents.")

        # Force HNSW compaction before process exits by running a test query
        test_vec = encoder.encode(["test"]).astype(float).tolist()
        collection.query(query_embeddings=test_vec, n_results=1)
        if verbose:
            print("HNSW index verified and ready.")
    else:
        if verbose:
            print(f"Collection '{collection_name}' already exists, skipping ingestion.")

    collection = client.get_or_create_collection(collection_name)
    return collection


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest product data into ChromaDB.")
    parser.add_argument("--lite", action="store_true", help="Use lite dataset")
    parser.add_argument("--db", default="products_vectorstore", help="ChromaDB directory name")
    parser.add_argument("--username", default="ed-donner", help="HuggingFace username")
    parser.add_argument("--collection", default="products", help="Collection name")
    args = parser.parse_args()

    ingest(
        lite_mode=args.lite,
        db=args.db,
        username=args.username,
        collection_name=args.collection,
    )
