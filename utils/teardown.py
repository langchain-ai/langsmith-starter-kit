"""LangSmith resource deletion helpers.

Usage in a use case's teardown:

    from utils.teardown import teardown_use_case

    teardown_use_case(
        project_name="starter-my-use-case",
        dataset_names=["My Dataset"],
        prompt_names=["my-eval-prompt"],
        queue_names=["My Review Queue"],
    )
"""
import requests
from typing import List, Optional

from utils.config import client, auth_headers, LANGSMITH_API_URL


def delete_project(project_name: str) -> None:
    for p in client.list_projects(name=project_name):
        client.delete_project(project_name=p.name)
        print(f"    - Deleted project: {p.name}")


def delete_datasets(names: List[str]) -> None:
    for name in names:
        for ds in client.list_datasets(dataset_name=name):
            client.delete_dataset(dataset_id=ds.id)
            print(f"    - Deleted dataset: {ds.name}")


def delete_prompts(names: List[str]) -> None:
    for name in names:
        try:
            client.delete_prompt(name)
            print(f"    - Deleted prompt: {name}")
        except Exception:
            pass


def delete_queues(names: List[str]) -> None:
    resp = requests.get(f"{LANGSMITH_API_URL}/annotation-queues", headers=auth_headers(), timeout=30)
    if resp.status_code >= 300:
        return
    for q in resp.json():
        if q.get("name") in names:
            r = requests.delete(f"{LANGSMITH_API_URL}/annotation-queues/{q['id']}", headers=auth_headers())
            print(f"    - Deleted queue: {q['name']} ({r.status_code})")


def _delete_taggings_for_resource(resource_id: str, resource_type: str) -> None:
    """Remove all Application tag taggings from a single resource (best-effort)."""
    base = f"{LANGSMITH_API_URL}/api/v1/workspaces/current"
    resp = requests.get(
        f"{base}/tags/resource",
        headers=auth_headers(),
        params={"resource_id": resource_id, "resource_type": resource_type},
        timeout=10,
    )
    if resp.status_code != 200:
        return
    for key_group in resp.json():
        for value in key_group.get("values", []):
            for tagging in value.get("taggings", []):
                requests.delete(
                    f"{base}/taggings/{tagging['id']}",
                    headers=auth_headers(), timeout=10,
                )


def _get_prompt_id(name: str) -> Optional[str]:
    resp = requests.get(f"{LANGSMITH_API_URL}/api/v1/repos/-/{name}", headers=auth_headers(), timeout=10)
    if resp.status_code == 200:
        return resp.json().get("repo", {}).get("id")
    return None


def delete_application_tags(
    project_name: str,
    dataset_names: List[str] = None,
    queue_names: List[str] = None,
    prompt_names: List[str] = None,
) -> None:
    """Remove all Application resource tag taggings from a use case's resources (best-effort)."""
    try:
        project = next(client.list_projects(name=project_name), None)
        if project:
            _delete_taggings_for_resource(str(project.id), "project")
            print(f"    - Cleared tags from project: {project_name}")

        for name in (dataset_names or []):
            ds = next(client.list_datasets(dataset_name=name), None)
            if ds:
                _delete_taggings_for_resource(str(ds.id), "dataset")
                print(f"    - Cleared tags from dataset: {name}")

        if queue_names:
            resp = requests.get(f"{LANGSMITH_API_URL}/annotation-queues", headers=auth_headers(), timeout=30)
            if resp.ok:
                for q in resp.json():
                    if q.get("name") in queue_names:
                        _delete_taggings_for_resource(q["id"], "queue")
                        print(f"    - Cleared tags from queue: {q['name']}")

        for name in (prompt_names or []):
            prompt_id = _get_prompt_id(name)
            if prompt_id:
                _delete_taggings_for_resource(prompt_id, "prompt")
                print(f"    - Cleared tags from prompt: {name}")
    except Exception:
        pass


def delete_tag_values(tags: List[str]) -> None:
    """Delete Application tag values by name, removing them from the workspace (best-effort)."""
    try:
        base = f"{LANGSMITH_API_URL}/api/v1/workspaces/current"
        resp = requests.get(f"{base}/tag-keys", headers=auth_headers(), timeout=10)
        if resp.status_code != 200:
            return
        app_key = next((k for k in resp.json() if k.get("key") == "Application"), None)
        if not app_key:
            return
        resp = requests.get(f"{base}/tag-keys/{app_key['id']}/tag-values", headers=auth_headers(), timeout=10)
        if resp.status_code != 200:
            return
        for val in resp.json():
            if val.get("value") in tags:
                r = requests.delete(
                    f"{base}/tag-keys/{app_key['id']}/tag-values/{val['id']}",
                    headers=auth_headers(), timeout=10,
                )
                if r.status_code in (200, 204):
                    print(f"    - Deleted tag value: {val['value']}")
    except Exception:
        pass


def teardown_use_case(
    project_name: str,
    dataset_names: List[str],
    prompt_names: List[str],
    queue_names: List[str],
    tags: Optional[List[str]] = None,
) -> None:
    """Delete all LangSmith resources for a use case."""
    print(f"Tearing down '{project_name}'...")
    delete_application_tags(project_name, dataset_names, queue_names, prompt_names)
    delete_project(project_name)
    delete_datasets(dataset_names)
    delete_prompts(prompt_names)
    delete_queues(queue_names)
    if tags:
        delete_tag_values(tags)
    print("Teardown complete.")
