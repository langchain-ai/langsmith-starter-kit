import os
import requests
import inspect
from dataclasses import dataclass
from typing import Dict, Optional, List, Callable, Any, Literal, Union

from setup.config import auth_headers, LANGSMITH_API_URL, client
from setup.prompts import prompt_exists



## Formatting Helpers
@dataclass
class JudgeEvaluator:
    name: str
    description: str
    score_type: Literal["boolean", "number", "string"]
    hub_ref: Optional[str] = None
    prompt: Optional[List[List[str]]] = None

@dataclass
class CodeEvaluator:
    code: Callable[..., Any]
    language: str = "python"

@dataclass
class EvaluatorPayload:
    display_name: str
    session_id: Optional[str] = None
    dataset_id: Optional[str] = None
    sampling_rate: float = 1.0
    is_enabled: bool = True
    filter: Optional[str] = None
    evaluators: Optional[List[JudgeEvaluator]] = None
    code_evaluators: Optional[List[CodeEvaluator]] = None


def _format_judge_evaluator(name: str, description: str, score_type: str, hub_ref: str = None, prompt: Optional[List[List[str]]] = None) -> Dict:
    payload = {
        "structured": {
            "model":{ 
                "lc":1,
                "type":"constructor",
                "id":["langchain","chat_models","openai","ChatOpenAI"],
                "kwargs":{
                    "temperature": 1,
                    "top_p": 1,
                    "presence_penalty": None,
                    "frequency_penalty": None,
                    "model":"gpt-5-mini",
                    "extra_headers":{},
                    "openai_api_key":{
                        "id":["OPENAI_API_KEY"],
                        "lc":1,
                        "type":"secret"
                    }
                },
            },
            "schema":{
                "title": "extract",
                "description": "Extract information from the user's response.",
                "type": "object",
                "properties": {
                    name: {
                        "type": score_type,
                        "description": description
                    },
                    "comment":{
                        "type": "string",
                        "description": "Reasoning for the score"
                    }
                },
                "required":[name]
            },
            "variable_mapping":{
                "input": "input",
                "output": "output",
                "reference": "referenceOutput"
            }
        }
    }
    if prompt:
        payload["structured"]["prompt"] = prompt
    elif hub_ref:
        payload["structured"]["hub_ref"] = hub_ref
    return payload

def _format_code_evaluator(code: Callable[..., Any], language: str = "python") -> Dict:
    payload = {
        "code": inspect.getsource(code),
        "language": language
    }
    return payload

## Validation Helpers
def evaluator_exists(name: str, target_type: Literal["dataset", "project"] = "dataset", target_id = None) -> bool:
    url = f"{LANGSMITH_API_URL}/api/v1/runs/rules"

    payload = {
        "dataset_id": target_id if target_type == "dataset" else None,
        "session_id": target_id if target_type == "project" else None,
        "name_contains": name
    }

    payload = {k: v for k, v in payload.items() if v is not None}
    existing = requests.get(url, headers=auth_headers(), params=payload, timeout=30)
    if existing.status_code >= 300:
        raise RuntimeError(f"Failed to search for evaluator '{name}': {existing.status_code} {existing.text}")
    
    existing = existing.json()
    for evaluator in existing:
        if evaluator.get("display_name") == name and (evaluator.get("evaluators") or evaluator.get("code_evaluators")):
            if target_type == "dataset" and evaluator.get("dataset_id") == target_id:
                return True
            elif target_type == "project" and evaluator.get("session_id") == target_id:
                return True
    return False


def _resolve_target_id(target_name: str, target_type: Literal["dataset", "project"] = "dataset") -> None:
    target = None
    if target_type == "dataset":
        datasets_iter = client.list_datasets(dataset_name=target_name)
        first_dataset = next(datasets_iter, None)
        if not first_dataset:
            print(f"    - Dataset '{target_name}' does not exist. Skipping evaluator...")
            return None
        target = first_dataset.id
    elif target_type == "project":
        projects_iter = client.list_projects(name=target_name)
        first_project = next(projects_iter, None)
        if not first_project:
            print(f"    - Project '{target_name}' does not exist. Skipping evaluator...")
            return None
        target = first_project.id
    return str(target)


## Evaluator Creation 
def create_judge_payload(name: str, prompt_or_ref: Union[str, List[List[str]]], sample_rate: float, score_type: Literal["boolean", "number", "string"], target_name: str, target_type: Literal["dataset", "project"] = "dataset", use_api: bool = False, owner: Optional[str] = None) -> None:
    target = _resolve_target_id(target_name, target_type)
    if not target:
        return None
    
    if evaluator_exists(name, target_type, target):
        print(f"    - Evaluator '{name}' already exists on the {target_type}. Skipping...")
        return None
    
    if isinstance(prompt_or_ref, list):
        pass
    else:
        if use_api:
            if not prompt_exists(str(prompt_or_ref), owner):
                print(f"    - Could not find {prompt_or_ref} via API. Skipping evaluator...")
                return None
        else:
            try:
                client.pull_prompt(prompt_or_ref)
            except Exception:
                print(f'    - Could not find {prompt_or_ref}. Skipping evaluator...')
                return None
          
    judge = _format_judge_evaluator(
        name=name,
        description=f"Evaluator for {name}",
        score_type=score_type,
        hub_ref=prompt_or_ref if isinstance(prompt_or_ref, str) else None,
        prompt=prompt_or_ref if isinstance(prompt_or_ref, list) else None,
    )
    
    payload = EvaluatorPayload(
        display_name=name,
        session_id=target if target_type == "project" else None,
        dataset_id=target if target_type == "dataset" else None,
        sampling_rate=sample_rate,
        filter="eq(is_root, true)",
        is_enabled=True,
        evaluators=[judge],
    )
    return payload


def create_code_payload(name: str, func: Callable[..., Any], language: Literal["python"], sample_rate: float, target_name: str, target_type: Literal["dataset", "project"] = "dataset") -> None:
    target = _resolve_target_id(target_name, target_type)
    if not target:
        return None
    
    if evaluator_exists(name, target_type, target):
        print(f"    - Evaluator '{name}' already exists on the {target_type}. Skipping...")
        return None

    code = _format_code_evaluator(func, language)

    payload = EvaluatorPayload(
        display_name=name,
        session_id=target if target_type == "project" else None,
        dataset_id=target if target_type == "dataset" else None,
        sampling_rate=sample_rate,
        filter="eq(is_root, true)",
        is_enabled=True,
        code_evaluators=[code],
    )
    return payload


def create_evaluator(payload: EvaluatorPayload) -> Optional[Dict]:
    url = f"{LANGSMITH_API_URL}/runs/rules"
    
    body = {
        "display_name": payload.display_name,
        "session_id": payload.session_id,
        "dataset_id": payload.dataset_id,
        "sampling_rate": payload.sampling_rate,
        "is_enabled": payload.is_enabled,
        "evaluators": payload.evaluators,
        "code_evaluators": payload.code_evaluators,
    }
    # Remove None fields to satisfy API schema
    body = {k: v for k, v in body.items() if v is not None}

    resp = requests.post(url, headers=auth_headers(), json=body, timeout=30)
    if resp.status_code >= 300:
        raise RuntimeError(f"Failed to create evaluator '{payload.display_name}': {resp.status_code} {resp.text}")
    return resp.json()


def load_evaluators(use_api: bool = False, owner: Optional[str] = None) -> None:
    """Create some example rules and attach them to our example datasets.

    Targets datasets created in setup/datasets.py via load_*_datasets functions.
    Uses prompts created in setup/prompts.py via load_*_prompt functions.
    """
    # Example rules. Adjust/extend as needed.
    # 1) Next action correctness rule (binary) via expression that expects a tool name in outputs
    #    This is illustrative; adapt the expression to your run schema.

    print(f"Creating evaluators...")
    # Dataset names from setup/datasets.py
    next_action_dataset = "Email Agent: Next Action"
    next_action_eval_ref = "email-agent-next-action-eval:latest"
    next_action_payload = create_judge_payload(
        name="correctness",
        prompt_or_ref=next_action_eval_ref,
        sample_rate=1.0,
        score_type="boolean",
        target_name=next_action_dataset,
        target_type="dataset",
        use_api=use_api,
        owner=owner,
    )


    final_response_dataset = "Email Agent: Final Response"
    final_response_eval_ref = "email-agent-final-response-eval:latest"
    final_response_payload = create_judge_payload(
        name="completeness",
        prompt_or_ref=final_response_eval_ref,
        sample_rate=1.0,
        score_type="boolean",
        target_name=final_response_dataset,
        target_type="dataset",
        use_api=use_api,
        owner=owner,
    )
    
    professionalism_project = os.getenv("LANGSMITH_PROJECT")
    professionalism_eval_ref = "email-agent-professionalism-eval:latest"
    professionalism_payload = create_judge_payload(
        name="professionalism",
        prompt_or_ref=professionalism_eval_ref,
        sample_rate=1.0,
        score_type="boolean",
        target_name=professionalism_project,
        target_type="project",
        use_api=use_api,
        owner=owner,
    )

    triage_dataset = "Email Agent: Triage"
    # Must be named perform_eval for code evaluators
    def perform_eval(run, example):
        correctness = example["outputs"]["classification"].lower() == run["outputs"]["output"]["content"].split("\n\n")[0].lower()
        return { "correctness": correctness }
    triage_code_payload = create_code_payload(
        name="triage_match",
        func=perform_eval,
        language="python",
        sample_rate=1.0,
        target_name=triage_dataset,
        target_type="dataset",
    )
    
    trajectory_dataset = "Email Agent: Trajectory"
    # Must be named perform_eval for code evaluators
    def perform_eval(run, example):
        """Evaluate whether the trajectory exactly matches the expected output"""
        return {
            "exact_match": run["outputs"]["trajectory"] == example["outputs"]["trajectory"]
        }
       
    trajectory_match_payload = create_code_payload(
        name="trajectory_match",
        func=perform_eval,
        language="python",
        sample_rate=1.0,
        target_name=trajectory_dataset,
        target_type="dataset",
    )

    evaluators = [
        next_action_payload, 
        final_response_payload, 
        professionalism_payload,
        triage_code_payload,
        trajectory_match_payload,
    ]

    for payload in evaluators:
        try:
            if payload:
                create_evaluator(payload)
                print(f"    - Evaluator '{payload.display_name}' created.")
        except Exception as e:
            print(f"    - Error creating evaluator '{payload.display_name}': {e}")
            continue


if __name__ == "__main__":
    load_evaluators()

