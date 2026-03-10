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
        _apply_project_tags(project_name, tags)


def _apply_project_tags(project_name: str, tags: List[str]) -> None:
    """Apply application tags to a LangSmith project via REST API."""
    try:
        url = f"{LANGSMITH_API_URL}/api/v1/sessions"
        resp = requests.get(url, headers=auth_headers(), params={"name": project_name}, timeout=30)
        if resp.status_code >= 300:
            return
        sessions = resp.json()
        if not sessions or not isinstance(sessions, list):
            return
        project_id = sessions[0].get("id")
        if not project_id:
            return
        tags_url = f"{LANGSMITH_API_URL}/api/v1/projects/{project_id}/tags"
        requests.post(tags_url, headers=auth_headers(), json={"tags": tags}, timeout=30)
    except Exception:
        pass  # Tags are best-effort


@traceable
def first_run(question: str) -> str:
    return "Hello, world!"