import csv
from pathlib import Path
from typing import List, Dict

from setup.config import client, auth_headers, LANGSMITH_API_URL
import requests


def _load_csv(path: Path) -> List[Dict]:
    """Load a CSV file and return list of row dicts."""
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _api_get_dataset_id(dataset_name: str):
    url = f"{LANGSMITH_API_URL}/api/v1/datasets"
    resp = requests.get(url, headers=auth_headers(), params={"name": dataset_name}, timeout=30)
    if resp.status_code >= 300:
        return None
    datasets = resp.json()
    if datasets and isinstance(datasets, list):
        first = datasets[0]
        if isinstance(first, dict) and "id" in first:
            return first["id"]
    return None


def _api_create_dataset(dataset_name: str, description: str = "") -> str:
    url = f"{LANGSMITH_API_URL}/api/v1/datasets"
    body = {"name": dataset_name, "description": description}
    resp = requests.post(url, headers=auth_headers(), json=body, timeout=30)
    if resp.status_code == 409:
        raise RuntimeError(f"Dataset '{dataset_name}' already exists")
    if resp.status_code >= 300:
        raise RuntimeError(f"Failed to create dataset '{dataset_name}': {resp.status_code} {resp.text}")
    data = resp.json()
    return data["id"]


def _api_create_examples(dataset_id: str, inputs: List[Dict], outputs: List[Dict]) -> None:
    url = f"{LANGSMITH_API_URL}/api/v1/examples/bulk"
    examples = [
        {"dataset_id": dataset_id, "inputs": inp, "outputs": out}
        for inp, out in zip(inputs, outputs)
    ]
    resp = requests.post(url, headers=auth_headers(), json=examples, timeout=60)
    if resp.status_code >= 300:
        raise RuntimeError(f"Failed to create examples: {resp.status_code} {resp.text}")


def _load_chatbot_dataset(dataset_name: str, csv_filename: str, use_api: bool = False) -> None:
    """Load a chatbot eval CSV into a LangSmith dataset."""
    data_path = Path(__file__).parent.parent / "data" / csv_filename
    rows = _load_csv(data_path)

    # inputs: question; outputs: answer + retrieved_chunks
    inputs = [{"question": row["question"]} for row in rows]
    outputs = [{"answer": row["answer"], "retrieved_chunks": row.get("retrieved_chunks", "")} for row in rows]

    if use_api:
        existing_id = _api_get_dataset_id(dataset_name)
        if existing_id:
            return
        dataset_id = _api_create_dataset(dataset_name)
        _api_create_examples(dataset_id, inputs, outputs)
    else:
        if not client.has_dataset(dataset_name=dataset_name):
            dataset = client.create_dataset(dataset_name=dataset_name)
            client.create_examples(
                inputs=inputs,
                outputs=outputs,
                dataset_id=dataset.id,
            )


def load_datasets(use_api: bool = False) -> None:
    print("Loading Datasets...")

    print("   - Chatbot: Ground Truth...")
    _load_chatbot_dataset(
        dataset_name="Chatbot: Ground Truth",
        csv_filename="eval_dataset_ground_truth.csv",
        use_api=use_api,
    )

    print("   - Chatbot: Multi-Source...")
    _load_chatbot_dataset(
        dataset_name="Chatbot: Multi-Source",
        csv_filename="eval_dataset_multi_source.csv",
        use_api=use_api,
    )


if __name__ == "__main__":
    load_datasets()
