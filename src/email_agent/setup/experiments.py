"""Email agent experiment — runs the agent on the trajectory dataset and scores it."""
import uuid
from typing import Any, List

from utils.config import client
from utils.datasets import api_get_dataset_id, api_list_examples
from utils.experiments import (
    api_create_session, api_create_run, api_end_run, api_log_feedback, api_close_session,
)
from src.email_agent.agent.agent import email_assistant


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_tool_calls(messages: List[Any]) -> List[str]:
    """Return the ordered list of tool names called in a message sequence."""
    names = []
    for msg in messages:
        calls = (
            msg.get("tool_calls") if isinstance(msg, dict)
            else getattr(msg, "tool_calls", None)
        )
        if calls:
            names.extend(call["name"].lower() for call in calls)
    return names


def _run_email_assistant(inputs: dict) -> dict:
    result = email_assistant.invoke(inputs, config={"thread_id": uuid.uuid4()})
    return {"trajectory": _extract_tool_calls(result["messages"])}


def _evaluate_extra_steps(outputs: dict, reference_outputs: dict) -> dict:
    """Count trajectory steps in outputs that are not in the reference."""
    ref = reference_outputs["trajectory"]
    out = outputs["trajectory"]
    i = j = unmatched = 0
    while i < len(ref) and j < len(out):
        if ref[i] == out[j]:
            i += 1
        else:
            unmatched += 1
        j += 1
    unmatched += len(out) - j
    return {"key": "unmatched_steps", "score": unmatched}


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def load_experiments(use_api: bool = False) -> None:
    print("Loading experiments...")
    print("     - Running trajectory experiment...")

    dataset = "Email Agent: Trajectory"

    if not use_api:
        client.evaluate(
            _run_email_assistant,
            data=dataset,
            evaluators=[_evaluate_extra_steps],
            experiment_prefix="email-agent-gpt4.1",
            num_repetitions=1,
            max_concurrency=4,
        )
    else:
        from datetime import datetime
        experiment_name = f"email-agent-gpt4.1-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        dataset_id = api_get_dataset_id(dataset)
        if not dataset_id:
            raise RuntimeError(f"Dataset '{dataset}' not found.")
        examples = api_list_examples(dataset_id)
        session_id = api_create_session(experiment_name, dataset_id)

        for ex in examples:
            inputs = ex.get("inputs", {})
            run_id = api_create_run(
                name="_run_email_assistant",
                inputs=inputs,
                session_id=session_id,
                reference_example_id=ex.get("id"),
            )
            outputs = _run_email_assistant(inputs)
            api_end_run(run_id, outputs)
            eval_result = _evaluate_extra_steps(outputs, ex.get("outputs", {}))
            try:
                api_log_feedback(run_id, eval_result["key"], eval_result["score"])
            except Exception as e:
                print(f"    - Warning: could not log feedback for run {run_id}: {e}")

        api_close_session(session_id)

    print("Experiments loaded successfully.")


if __name__ == "__main__":
    load_experiments()
