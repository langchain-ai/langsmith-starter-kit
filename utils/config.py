import os
import requests
from typing import Dict, List, Optional
from dotenv import load_dotenv
from langsmith import Client, traceable

load_dotenv(".env")

client = Client()

LANGSMITH_API_URL = os.getenv("LANGSMITH_API_URL", "https://api.smith.langchain.com")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_WORKSPACE_ID = os.getenv("LANGSMITH_WORKSPACE_ID")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def auth_headers() -> Dict[str, str]:
    if not LANGSMITH_API_KEY:
        raise RuntimeError("LANGSMITH_API_KEY is required in environment.")
    headers = {
        "x-api-key": LANGSMITH_API_KEY,
        "Content-Type": "application/json",
    }
    # For org-scoped API keys, include workspace header when provided
    if LANGSMITH_WORKSPACE_ID:
        headers["X-Tenant-ID"] = LANGSMITH_WORKSPACE_ID
    return headers

def setup_secrets() -> None:
    """
    Upsert workspace secrets (OPENAI_API_KEY) to LangSmith.
    """
    print("Setting up secrets...")
    if not OPENAI_API_KEY:
        print("    - OPENAI_API_KEY not set in environment; skipping workspace secret upsert.")
        return

    url = f"{LANGSMITH_API_URL}/workspaces/current/secrets"
    payload = [{"key": "OPENAI_API_KEY", "value": OPENAI_API_KEY}]

    resp = requests.post(url, headers=auth_headers(), json=payload, timeout=30)
    if resp.status_code >= 300:
        raise RuntimeError(f"Failed to upsert workspace secret OPENAI_API_KEY: {resp.status_code} {resp.text}")
    print(f"    - OPENAI_API_KEY upserted to LangSmith workspace.")

def setup_project(project_name: str, tags: Optional[List[str]] = None) -> None:
    os.environ["LANGSMITH_PROJECT"] = project_name
    first_run("Welcome to LangSmith!")
    if tags:
        project = next(client.list_projects(name=project_name), None)
        if project:
            apply_tags(str(project.id), "project", tags)


def apply_tags(resource_id: str, resource_type: str, tags: List[str]) -> None:
    """Apply Application resource tags to any LangSmith resource (best-effort).

    LangSmith Applications are workspace resource tags under the built-in
    "Application" tag key. Supported resource_type values: "project",
    "dataset", "annotation_queue".
    """
    try:
        base = f"{LANGSMITH_API_URL}/api/v1/workspaces/current"

        # Find (or create) the "Application" tag key
        resp = requests.get(f"{base}/tag-keys", headers=auth_headers(), timeout=10)
        if resp.status_code >= 300:
            return
        app_key = next((k for k in resp.json() if k.get("key") == "Application"), None)
        if not app_key:
            return
        app_key_id = app_key["id"]

        # Fetch existing tag values to avoid duplicates
        resp = requests.get(f"{base}/tag-keys/{app_key_id}/tag-values",
                            headers=auth_headers(), timeout=10)
        existing = {v["value"]: v["id"] for v in (resp.json() if resp.status_code == 200 else [])}

        # Fetch existing taggings on this resource to stay idempotent
        resp = requests.get(f"{base}/tags/resource",
                            headers=auth_headers(),
                            params={"resource_id": resource_id, "resource_type": resource_type},
                            timeout=10)
        applied_value_ids = {
            tagging["tag_value_id"]
            for key_group in (resp.json() if resp.status_code == 200 else [])
            for value in key_group.get("values", [])
            for tagging in value.get("taggings", [])
        }

        for tag in tags:
            # Get or create the tag value
            if tag in existing:
                tag_value_id = existing[tag]
            else:
                r = requests.post(f"{base}/tag-keys/{app_key_id}/tag-values",
                                  headers=auth_headers(), json={"value": tag}, timeout=10)
                if r.status_code >= 300:
                    continue
                tag_value_id = r.json()["id"]
                existing[tag] = tag_value_id

            if tag_value_id in applied_value_ids:
                continue

            requests.post(f"{base}/taggings", headers=auth_headers(),
                          json={"resource_id": resource_id, "resource_type": resource_type,
                                "tag_value_id": tag_value_id}, timeout=10)
    except Exception:
        pass  # Tags are best-effort


def _get_prompt_id(name: str) -> Optional[str]:
    """Return the repo ID for a prompt by name, or None if not found."""
    resp = requests.get(f"{LANGSMITH_API_URL}/api/v1/repos/-/{name}", headers=auth_headers(), timeout=10)
    if resp.status_code == 200:
        return resp.json().get("repo", {}).get("id")
    return None


def tag_all_resources(
    dataset_names: List[str],
    queue_names: List[str],
    prompt_names: List[str],
    tags: List[str],
) -> None:
    """Apply application tags to all use-case datasets, queues, and prompts."""
    for name in dataset_names:
        ds = next(client.list_datasets(dataset_name=name), None)
        if ds:
            apply_tags(str(ds.id), "dataset", tags)

    resp = requests.get(f"{LANGSMITH_API_URL}/annotation-queues", headers=auth_headers(), timeout=30)
    if resp.ok:
        for q in resp.json():
            if q.get("name") in queue_names:
                apply_tags(q["id"], "queue", tags)

    for name in prompt_names:
        prompt_id = _get_prompt_id(name)
        if prompt_id:
            apply_tags(prompt_id, "prompt", tags)


def get_project_id(name: str) -> Optional[str]:
    """Return the LangSmith project ID for the given project name."""
    first = next(client.list_projects(name=name), None)
    return str(first.id) if first else None


@traceable
def first_run(question: str) -> str:
    return "Hello, world!"
