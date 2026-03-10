"""Shared evaluator helpers for LangSmith.

Contributors can create evaluators with a single function call:

    # Code evaluator — write a perform_eval function and pass it:
    def perform_eval(run, example):
        return {"correctness": run["outputs"]["label"] == example["outputs"]["label"]}

    create_evaluator("correctness", "My Dataset", func=perform_eval)

    # LLM judge evaluator — inline prompt (no hub asset created):
    create_evaluator("quality", "My Dataset", prompt_or_ref=[
        ["system", "You are an expert evaluator..."],
        ["human", "Question: {input}\\n\\nAnswer: {output}\\n\\nGrade 0-1."],
    ], score_type="number")

    # LLM judge evaluator — push inline prompt to hub first, then reference it:
    create_evaluator("quality", "My Dataset", prompt_or_ref=[
        ["system", "You are an expert evaluator..."],
        ["human", "Question: {input}\\n\\nAnswer: {output}\\n\\nGrade 0-1."],
    ], score_type="number", push_prompt_as="my-quality-eval")

    # LLM judge evaluator — reference an existing hub prompt:
    create_evaluator("quality", "My Dataset", prompt_or_ref="my-prompt:latest", score_type="boolean")

All operations are idempotent — existing evaluators are skipped.
"""
import inspect
import textwrap
import requests
from typing import Callable, List, Literal, Optional, Union

from langchain_core.prompts.structured import StructuredPrompt

from utils.config import auth_headers, LANGSMITH_API_URL, client
from utils.prompts import load_prompt


def create_evaluator(
    name: str,
    target: str,
    target_type: Literal["dataset", "project"] = "dataset",
    *,
    func: Optional[Callable] = None,
    prompt_or_ref: Optional[Union[str, List]] = None,
    score_type: Literal["boolean", "number", "string"] = "boolean",
    sample_rate: float = 1.0,
    push_prompt_as: Optional[str] = None,
) -> None:
    """Create a LangSmith evaluator on a dataset or project.

    Provide exactly one of ``func`` (code evaluator) or ``prompt_or_ref``
    (LLM-judge evaluator).

    Args:
        name: Evaluator name, also used as the feedback key.
        target: Dataset name or project name.
        target_type: ``"dataset"`` or ``"project"``.
        func: Python callable for a code evaluator. The function may be named
            anything — it is automatically renamed to ``perform_eval`` when
            uploaded. All imports must be inside the function body.
        prompt_or_ref: Hub ref string (``"my-prompt:latest"``) or an inline
            prompt as ``[["system", "..."], ["human", "..."]]``.
        score_type: Score type for judge evaluators: ``"boolean"``,
            ``"number"``, or ``"string"``.
        sample_rate: Fraction of runs to evaluate (0.0–1.0).
        push_prompt_as: If provided alongside an inline ``prompt_or_ref``,
            pushes the prompt to LangSmith Hub under this name and wires the
            evaluator to reference it. The prompt becomes a reusable hub asset.
    """
    if (func is None) == (prompt_or_ref is None):
        raise ValueError("Provide exactly one of `func` or `prompt_or_ref`.")
    if push_prompt_as is not None and not isinstance(prompt_or_ref, list):
        raise ValueError("`push_prompt_as` requires an inline `prompt_or_ref` list.")

    target_id = _resolve_target_id(target, target_type)
    if not target_id:
        return

    if _evaluator_exists(name, target_type, target_id):
        print(f"    - Evaluator '{name}' already exists on {target_type} '{target}'. Skipping...")
        return

    if push_prompt_as and isinstance(prompt_or_ref, list):
        _push_eval_prompt(push_prompt_as, prompt_or_ref, name, score_type)
        prompt_or_ref = f"{push_prompt_as}:latest"

    if func is not None:
        body = _build_code_body(name, func, target_type, target_id, sample_rate)
    else:
        body = _build_judge_body(name, prompt_or_ref, score_type, target_type, target_id, sample_rate)

    if body is None:
        return

    url = f"{LANGSMITH_API_URL}/runs/rules"
    resp = requests.post(url, headers=auth_headers(), json=body, timeout=30)
    if resp.status_code >= 300:
        raise RuntimeError(f"Failed to create evaluator '{name}': {resp.status_code} {resp.text}")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _resolve_target_id(target: str, target_type: str) -> Optional[str]:
    if target_type == "dataset":
        first = next(client.list_datasets(dataset_name=target), None)
        if not first:
            print(f"    - Dataset '{target}' does not exist. Skipping evaluator...")
            return None
        return str(first.id)
    else:
        first = next(client.list_projects(name=target), None)
        if not first:
            print(f"    - Project '{target}' does not exist. Skipping evaluator...")
            return None
        return str(first.id)


def _evaluator_exists(name: str, target_type: str, target_id: str) -> bool:
    url = f"{LANGSMITH_API_URL}/api/v1/runs/rules"
    params = {
        ("dataset_id" if target_type == "dataset" else "session_id"): target_id,
        "name_contains": name,
    }
    resp = requests.get(url, headers=auth_headers(), params=params, timeout=30)
    if resp.status_code >= 300:
        raise RuntimeError(f"Failed to check evaluator '{name}': {resp.status_code} {resp.text}")
    key = "dataset_id" if target_type == "dataset" else "session_id"
    return any(
        e.get("display_name") == name
        and (e.get("evaluators") or e.get("code_evaluators"))
        and e.get(key) == target_id
        for e in resp.json()
    )


def _push_eval_prompt(prompt_name: str, messages: List, eval_name: str, score_type: str) -> None:
    """Push an inline eval prompt to LangSmith Hub as a named StructuredPrompt asset."""
    from src.model import eval_model
    schema = {
        "title": "extract",
        "description": "Extract information from the user's response.",
        "type": "object",
        "properties": {
            eval_name: {"type": score_type, "description": f"Evaluator score for {eval_name}"},
            "comment": {"type": "string", "description": "Reasoning for the score"},
        },
        "required": [eval_name, "comment"],
    }
    prompt = StructuredPrompt(messages=[tuple(m) for m in messages], schema_=schema)
    load_prompt(prompt_name, prompt, model=eval_model)


def _get_eval_source(func: Callable) -> str:
    """Serialize a callable to source, dedented and renamed to perform_eval."""
    source = textwrap.dedent(inspect.getsource(func))
    if func.__name__ != "perform_eval":
        source = source.replace(f"def {func.__name__}(", "def perform_eval(", 1)
    return source


def _build_code_body(name: str, func: Callable, target_type: str, target_id: str, sample_rate: float) -> dict:
    filter_key = "dataset_id" if target_type == "dataset" else "session_id"
    return {
        "display_name": name,
        filter_key: target_id,
        "sampling_rate": sample_rate,
        "is_enabled": True,
        "filter": "eq(is_root, true)",
        "code_evaluators": [{"code": _get_eval_source(func), "language": "python"}],
    }


def _build_judge_body(
    name: str,
    prompt_or_ref: Union[str, List],
    score_type: str,
    target_type: str,
    target_id: str,
    sample_rate: float,
) -> Optional[dict]:
    if isinstance(prompt_or_ref, str):
        try:
            client.pull_prompt(prompt_or_ref)
        except Exception:
            print(f"    - Prompt '{prompt_or_ref}' not found. Skipping evaluator...")
            return None

    structured = {
        "model": {
            "lc": 1, "type": "constructor",
            "id": ["langchain", "chat_models", "openai", "ChatOpenAI"],
            "kwargs": {
                "temperature": 1, "top_p": 1,
                "presence_penalty": None, "frequency_penalty": None,
                "model": "gpt-5-mini", "extra_headers": {},
                "openai_api_key": {"id": ["OPENAI_API_KEY"], "lc": 1, "type": "secret"},
            },
        },
        "schema": {
            "title": "extract",
            "description": "Extract information from the user's response.",
            "type": "object",
            "properties": {
                name: {"type": score_type, "description": f"Evaluator score for {name}"},
                "comment": {"type": "string", "description": "Reasoning for the score"},
            },
            "required": [name, "comment"],
        },
        "variable_mapping": {"input": "input", "output": "output", "reference": "referenceOutput"},
    }
    if isinstance(prompt_or_ref, list):
        structured["prompt"] = prompt_or_ref
    else:
        structured["hub_ref"] = prompt_or_ref

    filter_key = "dataset_id" if target_type == "dataset" else "session_id"
    return {
        "display_name": name,
        filter_key: target_id,
        "sampling_rate": sample_rate,
        "is_enabled": True,
        "filter": "eq(is_root, true)",
        "evaluators": [{"structured": structured}],
    }
