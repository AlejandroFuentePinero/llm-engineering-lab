"""
Entrypoint for the GPT-5.1 + RAG price prediction benchmark.

Runs rag_pipeline() which evaluates GPT-5.1 augmented with retrieval from the
ChromaDB vectorstore of 800k product embeddings. For each test item, the 5 most
similar products and their prices are retrieved and injected as context into the
prompt before the LLM call.

Requires the vectorstore to be built first:
    uv run python src/pricer/RAG/rag_ingest.py
"""

# --- Paths
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.pricer.RAG.rag_pipeline import rag_pipeline


if __name__ == "__main__":
    rag_pipeline()
