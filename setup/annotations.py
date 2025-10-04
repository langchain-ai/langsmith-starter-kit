import requests
from dataclasses import dataclass
from typing import Optional, List, Dict

from setup.config import auth_headers, LANGSMITH_API_URL, LANGSMITH_PROJECT, client    

@dataclass
class AutomationPayload:
    display_name: str
    session_id: str
    annotation_queue_id: str
    filter: str
    is_enabled: bool = True
    sampling_rate: float = 1.0

@dataclass
class QueuePayload:
    name: str
    enable_reservations: bool = True
    num_reviewers_per_item: int = 1
    reservation_minutes: int = 1
    description: Optional[str] = None
    instructions: Optional[str] = None
    rubric_items: Optional[List[Dict]] = None

def _get_project_id(name: str) -> str:
    projects = client.list_projects()
    for project in projects:
        if project.name == name:
            return project.id
    return None

def automation_exists(name: str, project_id = None) -> bool:
    url = f"{LANGSMITH_API_URL}/api/v1/runs/rules"

    payload = {
        "session_id": project_id,
        "name_contains": name
    }

    payload = {k: v for k, v in payload.items() if v is not None}
    existing = requests.get(url, headers=auth_headers(), params=payload, timeout=30)
    if existing.status_code >= 300:
        raise RuntimeError(f"Failed to search for evaluator '{name}': {existing.status_code} {existing.text}")
    
    existing = existing.json()
    for evaluator in existing:
        if evaluator.get("display_name") == name and (evaluator.get("evaluators") or evaluator.get("code_evaluators")):
            if evaluator.get("session_id") == project_id:
                return True
    return False

def get_queue_id(name: str) -> str:
    url = f"{LANGSMITH_API_URL}/annotation-queues"
    resp = requests.get(url, headers=auth_headers(), timeout=30)
    if resp.status_code >= 300:
        raise RuntimeError(f"Failed to search for queue '{name}': {resp.status_code} {resp.text}")
    resp = resp.json()
    for queue in resp:
        if queue.get("name") == name:
            return queue.get("id")
    return None


def create_automation(automation: AutomationPayload) -> None:
    payload = {
        "sampling_rate": automation.sampling_rate,
        "is_enabled": True,
        "add_to_annotation_queue_id": automation.annotation_queue_id,
        "display_name": automation.display_name,
        "session_id": automation.session_id,
        "filter": automation.filter,
    }

    url = f"{LANGSMITH_API_URL}/runs/rules"
    resp = requests.post(url, headers=auth_headers(), json=payload, timeout=30)
    if resp.status_code >= 300:
        raise RuntimeError(f"Failed to create automation '{automation.display_name}': {resp.status_code} {resp.text}")
    return resp.json()

def create_queue(queue: QueuePayload) -> str:
    payload = {
        "name": queue.name,
        "enable_reservations": queue.enable_reservations,
        "num_reviewers_per_item": queue.num_reviewers_per_item,
        "reservation_minutes": queue.reservation_minutes,
        "rubric_items": queue.rubric_items,
    }

    url = f"{LANGSMITH_API_URL}/annotation-queues"
    resp = requests.post(url, headers=auth_headers(), json=payload, timeout=30)
    if resp.status_code >= 300:
        raise RuntimeError(f"Failed to create annotation queue '{queue.name}': {resp.status_code} {resp.text}")
    data = resp.json()
    return data.get("id")

def load_automations_and_queues() -> None:
    project_name = LANGSMITH_PROJECT
    project_id = _get_project_id(project_name)

    queue_payload = QueuePayload(
        name="Professionalism Annotation Queue",
        description="Queue for manual review of professional responses",
        instructions="Please review the response and determine if it is professional or not.",
        rubric_items=[
            {"feedback_key": "professionalism"},
        ],
    )
    queues = [queue_payload]

    
    print(f"Creating queues and automations...")
    for queue in queues:

        queue_id = get_queue_id(queue.name)
        if not queue_id:
            queue_id = create_queue(queue)
            print(f"     - Queue '{queue.name}' created.")
        else:
            print(f"     - Queue '{queue.name}' already exists, using id...")

        if not project_id:
            print(f"Project '{project_name}' not found, skipping automation for this queue...")
            continue
        
        queue_rule_payload = AutomationPayload(
            display_name="Professional Review",
            session_id=str(project_id),
            annotation_queue_id=str(queue_id),
            filter="and(eq(is_root, true), and(eq(feedback_key, \"professionalism\"), eq(feedback_score, 1)))",
            sampling_rate=1.0,
        )
        if automation_exists(queue_rule_payload.display_name, project_id):
            print(f"Automation '{queue_rule_payload.display_name}' already exists, skipping...")
            continue
        
        create_automation(queue_rule_payload)
        print(f"     - Automation '{queue_rule_payload.display_name}' created for queue '{queue.name}'.")

if __name__ == "__main__":
    load_automations_and_queues()