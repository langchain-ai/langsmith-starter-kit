"""Email agent annotation queues and automations."""
import os

from utils.config import get_project_id
from utils.annotations import get_queue_id, create_queue, automation_exists, create_automation


def load_automations_and_queues() -> None:
    project_name = os.getenv("LANGSMITH_PROJECT")
    project_id = get_project_id(project_name)

    print("Creating queues and automations...")

    queue_name = "Professionalism Annotation Queue"
    queue_id = get_queue_id(queue_name)
    if not queue_id:
        queue_id = create_queue(
            name=queue_name,
            description="Queue for manual review of professional responses",
            instructions="Please review the response and determine if it is professional or not.",
            rubric_items=[{"feedback_key": "professionalism"}],
        )
        print(f"     - Queue '{queue_name}' created.")
    else:
        print(f"     - Queue '{queue_name}' already exists.")

    if not project_id:
        print(f"Project '{project_name}' not found, skipping automation...")
        return

    automation_name = "Professional Review"
    if automation_exists(automation_name, project_id):
        print(f"     - Automation '{automation_name}' already exists. Skipping...")
        return

    create_automation(
        name=automation_name,
        project_id=str(project_id),
        queue_id=str(queue_id),
        filter='and(eq(is_root, true), and(eq(feedback_key, "professionalism"), eq(feedback_score, 1)))',
        sampling_rate=1.0,
    )
    print(f"     - Automation '{automation_name}' created for queue '{queue_name}'.")


if __name__ == "__main__":
    load_automations_and_queues()
