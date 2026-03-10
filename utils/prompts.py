"""Shared prompt helpers for LangSmith (SDK + REST API).

Usage:
    load_prompt("my-prompt", my_chain, use_api=False)
    delete_existing_prompt("my-prompt", use_api=False)
    build_schema(MyPydanticModel, "field_name")
"""
import json
import requests
from typing import Any, Optional

from langsmith.utils import LangSmithConflictError
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.structured import StructuredPrompt
from langchain_core.runnables import RunnableBinding, RunnableSequence
from langchain_core.load.dump import dumps
from pydantic import BaseModel

from utils.config import auth_headers, LANGSMITH_API_URL, client


# ---------------------------------------------------------------------------
# Workspace / hub helpers
# ---------------------------------------------------------------------------

def get_owner(owner: Optional[str] = None) -> str:
    """Resolve the workspace owner handle. Personal workspaces use '-'."""
    if owner:
        return owner
    resp = requests.get(f"{LANGSMITH_API_URL}/api/v1/settings", headers=auth_headers(), timeout=30)
    if resp.status_code >= 300:
        raise RuntimeError(f"Failed to fetch settings: {resp.status_code} {resp.text}")
    return resp.json().get("tenant_handle") or "-"


def prompt_exists(prompt_or_ref: str, owner: Optional[str] = None) -> bool:
    """Return True if the prompt repo/commit exists via the LangSmith API."""
    resolved_owner = get_owner(owner)
    repo, version = (prompt_or_ref.split(":", 1) if ":" in prompt_or_ref else (prompt_or_ref, None))
    try:
        if version is None or version == "latest":
            url = f"{LANGSMITH_API_URL}/api/v1/commits/{resolved_owner}/{repo}/latest"
        else:
            url = f"{LANGSMITH_API_URL}/api/v1/commits/{resolved_owner}/{repo}/{version}"
        resp = requests.get(url, headers=auth_headers(), timeout=20)
        if resp.status_code == 200:
            return True
        if resp.status_code == 404:
            return False
        # Fallback: check repo commits list
        list_resp = requests.get(
            f"{LANGSMITH_API_URL}/api/v1/repos/{resolved_owner}/{repo}/commits",
            headers=auth_headers(), timeout=20,
        )
        return list_resp.status_code == 200 and bool(list_resp.json())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Push / delete helpers
# ---------------------------------------------------------------------------

def prep_runnable_for_push(obj: Any) -> Any:
    """Normalize a (ChatPromptTemplate | model) chain for API upload."""
    if not (
        isinstance(obj, RunnableSequence)
        and isinstance(obj.first, ChatPromptTemplate)
        and len(obj.steps) > 1
        and isinstance(obj.steps[1], RunnableBinding)
        and len(obj.steps) <= 3
    ):
        return obj

    prompt = obj.first
    bound_model = obj.steps[1]
    model_kwargs = bound_model.kwargs

    if (
        not isinstance(prompt, StructuredPrompt)
        and isinstance(model_kwargs, dict)
        and "ls_structured_output_format" in model_kwargs
    ):
        output_format = model_kwargs["ls_structured_output_format"]
        prompt = StructuredPrompt(messages=prompt.messages, **output_format)

    if isinstance(prompt, StructuredPrompt):
        try:
            structured_kwargs = (prompt | bound_model.bound).steps[1].kwargs
        except Exception:
            structured_kwargs = {}
        bound_model.kwargs = {k: v for k, v in (model_kwargs or {}).items() if k not in (structured_kwargs or {})}
        return RunnableSequence(prompt, bound_model)

    return obj


def api_push_prompt_commit(name: str, obj: Any, owner: Optional[str] = None) -> Optional[str]:
    """Push a prompt commit via the LangSmith REST API."""
    resolved_owner = get_owner(owner)
    url = f"{LANGSMITH_API_URL}/api/v1/commits/{resolved_owner}/{name}"
    prepped = prep_runnable_for_push(obj)

    try:
        from langchain_core.load.dump import dumpd
        manifest = dumpd(prepped)
    except Exception:
        try:
            manifest = json.loads(dumps(prepped))
        except Exception as e:
            raise RuntimeError(f"Failed to serialize prompt '{name}': {e}")

    body: dict = {"manifest": manifest}
    try:
        latest = requests.get(f"{url}/latest", headers=auth_headers(), timeout=15)
        if latest.status_code == 200:
            parent = latest.json().get("commit_hash")
            if parent:
                body["parent_commit"] = parent
    except Exception:
        pass

    resp = requests.post(url, headers=auth_headers(), json=body, timeout=30)
    if resp.status_code == 409:
        return None  # Unchanged — treat as no-op
    if resp.status_code == 404:
        # Repo doesn't exist yet — create it, then retry
        create = requests.post(
            f"{LANGSMITH_API_URL}/api/v1/repos",
            headers=auth_headers(),
            json={"repo_handle": name, "owner_handle": resolved_owner, "is_public": False},
            timeout=30,
        )
        if create.status_code not in (200, 201, 409):
            raise RuntimeError(f"Failed to create prompt repo: {create.status_code} {create.text}")
        retry = requests.post(url, headers=auth_headers(), json=body, timeout=30)
        if retry.status_code == 409:
            return None
        if retry.status_code >= 300:
            raise RuntimeError(f"Failed to push prompt: {retry.status_code} {retry.text}")
        return f"{LANGSMITH_API_URL}/repos/{resolved_owner}/{name}"
    if resp.status_code >= 300:
        raise RuntimeError(f"Failed to push prompt: {resp.status_code} {resp.text}")
    return f"{LANGSMITH_API_URL}/repos/{resolved_owner}/{name}"


def api_delete_prompt_repo(name: str, owner: Optional[str] = None) -> None:
    """Delete a prompt repo via the LangSmith REST API (best-effort)."""
    resolved_owner = get_owner(owner)
    try:
        resp = requests.delete(
            f"{LANGSMITH_API_URL}/api/v1/repos/{resolved_owner}/{name}",
            headers=auth_headers(), timeout=30,
        )
        if resp.status_code not in (200, 204, 404):
            raise RuntimeError(f"Failed to delete prompt '{name}': {resp.status_code} {resp.text}")
    except Exception:
        pass


def load_prompt(name: str, obj: Any, use_api: bool = False, owner: Optional[str] = None) -> Optional[str]:
    """Push a prompt commit using the SDK (preferred) or REST API."""
    if use_api:
        return api_push_prompt_commit(name=name, obj=obj, owner=owner)
    try:
        return client.push_prompt(name, object=obj)
    except LangSmithConflictError:
        return None  # Unchanged since last commit


def delete_existing_prompt(name: str, use_api: bool = False, owner: Optional[str] = None) -> None:
    """Delete a prompt by name (best-effort, used before recreating)."""
    if use_api:
        api_delete_prompt_repo(name=name, owner=owner)
        print(f"    ...deleted existing prompt (api): {name}")
    else:
        try:
            client.delete_prompt(name)
            print(f"    ...deleted existing prompt: {name}")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Schema helper for StructuredPrompt evaluators
# ---------------------------------------------------------------------------

def build_schema(model: type[BaseModel], name: str) -> dict:
    """Build a JSON schema from a Pydantic model for use in StructuredPrompt."""
    schema = model.model_json_schema()
    schema["description"] = "Extract information from the user's response."
    schema["title"] = "extract"
    schema["properties"][name].pop("title", None)
    return schema
