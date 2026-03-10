"""Shared annotation queue and automation helpers for LangSmith."""
import requests
from typing import Dict, List, Optional

from utils.config import auth_headers, LANGSMITH_API_URL


def get_queue_id(name: str) -> Optional[str]:
    """Return the ID of an annotation queue by name, or None if not found."""
    resp = requests.get(f"{LANGSMITH_API_URL}/annotation-queues", headers=auth_headers(), timeout=30)
    if resp.status_code >= 300:
        raise RuntimeError(f"Failed to list queues: {resp.status_code} {resp.text}")
    return next((q["id"] for q in resp.json() if q.get("name") == name), None)


def automation_exists(name: str, project_id: Optional[str] = None) -> bool:
    """Return True if an automation with the given display name already exists."""
    params = {k: v for k, v in {"session_id": project_id, "name_contains": name}.items() if v is not None}
    resp = requests.get(f"{LANGSMITH_API_URL}/api/v1/runs/rules", headers=auth_headers(), params=params, timeout=30)
    if resp.status_code >= 300:
        raise RuntimeError(f"Failed to check automation '{name}': {resp.status_code} {resp.text}")
    return any(a.get("display_name") == name for a in resp.json())


def create_queue(
    name: str,
    description: Optional[str] = None,
    instructions: Optional[str] = None,
    rubric_items: Optional[List[Dict]] = None,
    enable_reservations: bool = True,
    num_reviewers_per_item: int = 1,
    reservation_minutes: int = 1,
) -> str:
    """Create an annotation queue and return its ID."""
    payload = {
        "name": name,
        "enable_reservations": enable_reservations,
        "num_reviewers_per_item": num_reviewers_per_item,
        "reservation_minutes": reservation_minutes,
        "rubric_items": rubric_items,
    }
    resp = requests.post(f"{LANGSMITH_API_URL}/annotation-queues", headers=auth_headers(), json=payload, timeout=30)
    if resp.status_code >= 300:
        raise RuntimeError(f"Failed to create queue '{name}': {resp.status_code} {resp.text}")
    return resp.json()["id"]


def create_automation(
    name: str,
    project_id: str,
    queue_id: str,
    filter: str,
    sampling_rate: float = 1.0,
) -> None:
    """Create a run automation that routes matching traces into an annotation queue."""
    payload = {
        "display_name": name,
        "session_id": project_id,
        "add_to_annotation_queue_id": queue_id,
        "filter": filter,
        "sampling_rate": sampling_rate,
        "is_enabled": True,
    }
    resp = requests.post(f"{LANGSMITH_API_URL}/runs/rules", headers=auth_headers(), json=payload, timeout=30)
    if resp.status_code >= 300:
        raise RuntimeError(f"Failed to create automation '{name}': {resp.status_code} {resp.text}")
