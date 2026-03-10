"""Shared dataset helpers for LangSmith.

The API helpers (api_get_dataset_id, api_list_examples) are kept here because
they are used by utils/experiments.py for the API-only experiment path.
"""
import requests
from typing import Any, Dict, List, Optional

from utils.config import client, auth_headers, LANGSMITH_API_URL


def create_langsmith_dataset(
    name: str,
    inputs: List[Dict],
    outputs: List[Dict],
    description: str = "",
) -> None:
    """Create a LangSmith dataset with examples if it doesn't already exist."""
    if not client.has_dataset(dataset_name=name):
        dataset = client.create_dataset(dataset_name=name, description=description)
        client.create_examples(inputs=inputs, outputs=outputs, dataset_id=dataset.id)


# ---------------------------------------------------------------------------
# REST API helpers — used by utils/experiments.py for API-only experiments
# ---------------------------------------------------------------------------

def api_get_dataset_id(dataset_name: str) -> Optional[str]:
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


def api_list_examples(dataset_id: str) -> List[Dict[str, Any]]:
    url = f"{LANGSMITH_API_URL}/api/v1/examples"
    resp = requests.get(url, headers=auth_headers(), params={"dataset": dataset_id}, timeout=30)
    if resp.status_code >= 300:
        raise RuntimeError(f"Failed to list examples for dataset '{dataset_id}': {resp.status_code} {resp.text}")
    items = resp.json()
    return items if isinstance(items, list) else []


def _api_create_dataset(dataset_name: str, description: str = "") -> str:
    url = f"{LANGSMITH_API_URL}/api/v1/datasets"
    body = {"name": dataset_name, "description": description}
    resp = requests.post(url, headers=auth_headers(), json=body, timeout=30)
    if resp.status_code == 409:
        raise RuntimeError(f"Dataset '{dataset_name}' already exists")
    if resp.status_code >= 300:
        raise RuntimeError(f"Failed to create dataset '{dataset_name}': {resp.status_code} {resp.text}")
    return resp.json()["id"]


def _api_create_examples(dataset_id: str, inputs: List[Dict], outputs: List[Dict]) -> None:
    url = f"{LANGSMITH_API_URL}/api/v1/examples/bulk"
    examples = [
        {"dataset_id": dataset_id, "inputs": inp, "outputs": out}
        for inp, out in zip(inputs, outputs)
    ]
    resp = requests.post(url, headers=auth_headers(), json=examples, timeout=60)
    if resp.status_code >= 300:
        raise RuntimeError(f"Failed to create examples in bulk: {resp.status_code} {resp.text}")
