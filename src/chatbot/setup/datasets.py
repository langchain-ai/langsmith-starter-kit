"""Chatbot dataset setup — loads evaluation CSVs into LangSmith datasets."""
import csv
from pathlib import Path
from typing import Dict, List

from utils.datasets import create_langsmith_dataset

_DATA_DIR = Path(__file__).parent.parent / "data"


def _load_csv(filename: str) -> List[Dict]:
    with open(_DATA_DIR / filename, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_datasets(use_api: bool = False) -> None:
    print("Loading Datasets...")

    for dataset_name, csv_file in [
        ("Chatbot: Ground Truth", "eval_dataset_ground_truth.csv"),
        ("Chatbot: Multi-Source", "eval_dataset_multi_source.csv"),
    ]:
        print(f"   - {dataset_name}...")
        rows = _load_csv(csv_file)
        create_langsmith_dataset(
            dataset_name,
            inputs=[{"question": r["question"]} for r in rows],
            outputs=[{"answer": r["answer"], "retrieved_chunks": r.get("retrieved_chunks", "")} for r in rows],
            use_api=use_api,
        )


if __name__ == "__main__":
    load_datasets()
