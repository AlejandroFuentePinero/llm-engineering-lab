# --- Paths
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

# --- Imports
from src.pricer.agents.autonomous_planning_agent import AutonomousPlanningAgent
import chromadb
import logging


def run_agent(db_path: str = "products_vectorstore", db_name: str = "products"):

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    client = chromadb.PersistentClient(path=str(project_root / db_path))
    collection = client.get_or_create_collection(db_name)
    agent = AutonomousPlanningAgent(collection)
    agent.plan()


if __name__ == "__main__":
    run_agent()
