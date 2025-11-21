import uuid
from typing import Any, List, Dict, Optional, Tuple
from datetime import datetime
import requests

from setup.config import client, auth_headers, LANGSMITH_API_URL
from langsmith.run_helpers import get_run_tree_context
from setup.datasets import api_get_dataset_id, api_list_examples
from agent.agent import email_assistant

# Helper to extract tool trajectory
def extract_tool_calls(messages: List[Any]) -> List[str]:
    """Extract tool call names from messages, safely handling messages without tool_calls."""
    tool_call_names = []
    for message in messages:
        # Check if message is a dict and has tool_calls
        if isinstance(message, dict) and message.get("tool_calls"):
            tool_call_names.extend([call["name"].lower() for call in message["tool_calls"]])
        # Check if message is an object with tool_calls attribute
        elif hasattr(message, "tool_calls") and message.tool_calls:
            tool_call_names.extend([call["name"].lower() for call in message.tool_calls])
    
    return tool_call_names

# Define Run Function for your Application
def run_email_assistant(inputs: dict) -> dict:
    """Run the email assistant on the given email input."""
    # Creating configuration 
    thread_id = uuid.uuid4()
    configuration = {"thread_id": thread_id}

    result = email_assistant.invoke(inputs, config = configuration)
    return {"trajectory": extract_tool_calls(result["messages"])}


# Define Evaluator Functions
def evaluate_extra_steps(outputs: dict, reference_outputs: dict) -> dict:
    """Evaluate the number of unmatched steps in the agent's output."""
    i = j = 0
    unmatched_steps = 0

    while i < len(reference_outputs['trajectory']) and j < len(outputs['trajectory']):
        if reference_outputs['trajectory'][i] == outputs['trajectory'][j]:
            i += 1  # Match found, move to the next step in reference trajectory
        else:
            unmatched_steps += 1  # Step is not part of the reference trajectory
        j += 1  # Always move to the next step in outputs trajectory

    # Count remaining unmatched steps in outputs beyond the comparison loop
    unmatched_steps += len(outputs['trajectory']) - j

    return {
        "key": "unmatched_steps",
        "score": unmatched_steps,
    }

# -------------------------
# API Helpers (optional)
# -------------------------

def api_create_session(name: str, dataset_id: str) -> str:
    """Create an experiment session linked to a dataset (appears in Experiments)."""
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

def api_log_feedback(run_id: str, key: str, score: Any, comment: Optional[str] = None) -> None:
    """Log feedback incrementally for a run using POST /api/v1/feedback."""
    url = f"{LANGSMITH_API_URL}/api/v1/feedback"
    body = {
        "run_id": run_id,
        "key": key,
        "score": score,
        "comment": comment or "",
    }
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

def api_create_run(name: str, inputs: Dict[str, Any], session_id: str, reference_example_id: Optional[str], run_type: str = "chain") -> str:
    """Create a root run explicitly linked to a dataset example."""
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
    # API returns a message, not the full object; return the generated id
    return run_id

def api_end_run(run_id: str, outputs: Dict[str, Any]) -> None:
    """End a run by updating it with an end_time (and outputs)."""
    url = f"{LANGSMITH_API_URL}/api/v1/runs/{run_id}"
    body = {
        "end_time": datetime.utcnow().isoformat() + "Z",
        "outputs": outputs,
    }
    resp = requests.patch(url, headers=auth_headers(), json=body, timeout=60)
    if resp.status_code >= 300:
        raise RuntimeError(f"Failed to end run '{run_id}': {resp.status_code} {resp.text}")


# Run Experiment
def run_trajectory_experiment(use_api: bool = False) -> dict:   
    trajectory_dataset = "Email Agent: Trajectory"
    if not use_api:
        results = client.evaluate(
            run_email_assistant,
            data=trajectory_dataset,
            evaluators=[evaluate_extra_steps],
            experiment_prefix="email-agent-gpt4.1",
            num_repetitions=1,
            max_concurrency=4,
        )
        return results
    else:
        # API path: create a session, iterate examples, create/end runs, attach feedback
        experiment_name = f"email-agent-gpt4.1-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        dataset_id = api_get_dataset_id(trajectory_dataset)
        if not dataset_id:
            raise RuntimeError(f"Dataset '{trajectory_dataset}' not found.")
        examples = api_list_examples(dataset_id)
        session_id = api_create_session(experiment_name, dataset_id)
        
        run_summaries: List[Dict[str, Any]] = []
        for ex in examples:
            inputs = ex.get("inputs", {})
            reference_outputs = ex.get("outputs", {})
            example_id = ex.get("id")
            # Create a root run explicitly linked to this dataset example
            run_id = api_create_run(
                name="run_email_assistant",
                inputs=inputs,
                session_id=session_id,
                reference_example_id=example_id,
            )
            # Execute the graph (auto-tracing will attach child runs)
            outputs = run_email_assistant(inputs)
            # End the root run with outputs
            api_end_run(run_id, outputs)
            # Evaluate and log feedback incrementally
            eval_res = evaluate_extra_steps(outputs, reference_outputs)
            try:
                api_log_feedback(
                    run_id=run_id,
                    key=eval_res["key"],
                    score=eval_res["score"],
                    comment=f"reference={reference_outputs}"
                )
            except Exception as e:
                print(f"    - Warning: could not log feedback for run {run_id}: {e}")
            run_summaries.append({
                "run_id": run_id,
                "score_key": eval_res["key"],
                "score": eval_res["score"],
            })
        # Close the experiment session
        api_close_session(session_id)
        return {
            "session_id": session_id,
            "experiment_name": experiment_name,
            "runs": run_summaries,
        }


def load_experiments(use_api: bool = False) -> None:
    print("Loading experiments...")
    print("     - Running trajectory experiment...")
    run_trajectory_experiment(use_api=use_api)
    print("Experiments loaded successfully.")

if __name__ == "__main__":
    load_experiments()