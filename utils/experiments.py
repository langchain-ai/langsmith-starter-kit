"""Shared experiment/session helpers for the LangSmith REST API."""
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import requests

from utils.config import auth_headers, LANGSMITH_API_URL


def api_create_session(name: str, dataset_id: str) -> str:
    """Create an experiment session linked to a dataset."""
    url = f"{LANGSMITH_API_URL}/api/v1/sessions"
    body = {
        "name": name,
        "start_time": datetime.utcnow().isoformat() + "Z",
        "reference_dataset_id": dataset_id,
    }
    resp = requests.post(url, headers=auth_headers(), json=body, timeout=30)
    if resp.status_code >= 300:
        raise RuntimeError(f"Failed to create session '{name}': {resp.status_code} {resp.text}")
    return resp.json()["id"]


def api_create_run(
    name: str,
    inputs: Dict[str, Any],
    session_id: str,
    reference_example_id: Optional[str],
    run_type: str = "chain",
) -> str:
    """Create a root run linked to a dataset example."""
    url = f"{LANGSMITH_API_URL}/api/v1/runs"
    run_id = str(uuid.uuid4())
    body = {
        "id": run_id,
        "name": name,
        "inputs": inputs,
        "session_id": session_id,
        "reference_example_id": reference_example_id,
        "start_time": datetime.utcnow().isoformat() + "Z",
        "run_type": run_type,
        "is_root": True,
    }
    resp = requests.post(url, headers=auth_headers(), json=body, timeout=60)
    if resp.status_code >= 300:
        raise RuntimeError(f"Failed to create run: {resp.status_code} {resp.text}")
    return run_id


def api_end_run(run_id: str, outputs: Dict[str, Any]) -> None:
    """End a run by patching it with outputs and end_time."""
    url = f"{LANGSMITH_API_URL}/api/v1/runs/{run_id}"
    body = {"end_time": datetime.utcnow().isoformat() + "Z", "outputs": outputs}
    resp = requests.patch(url, headers=auth_headers(), json=body, timeout=60)
    if resp.status_code >= 300:
        raise RuntimeError(f"Failed to end run '{run_id}': {resp.status_code} {resp.text}")


def api_log_feedback(
    run_id: str, key: str, score: Any, comment: Optional[str] = None
) -> None:
    """Log a feedback score for a run."""
    url = f"{LANGSMITH_API_URL}/api/v1/feedback"
    body = {"run_id": run_id, "key": key, "score": score, "comment": comment or ""}
    resp = requests.post(url, headers=auth_headers(), json=body, timeout=30)
    if resp.status_code >= 300:
        raise RuntimeError(f"Failed to log feedback for run '{run_id}': {resp.status_code} {resp.text}")


def api_close_session(session_id: str) -> None:
    """Close an experiment session by setting end_time."""
    url = f"{LANGSMITH_API_URL}/api/v1/sessions/{session_id}"
    body = {"end_time": datetime.utcnow().isoformat() + "Z"}
    resp = requests.patch(url, headers=auth_headers(), json=body, timeout=30)
    if resp.status_code >= 300:
        raise RuntimeError(f"Failed to close session '{session_id}': {resp.status_code} {resp.text}")
