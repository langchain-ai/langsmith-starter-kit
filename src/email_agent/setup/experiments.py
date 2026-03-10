"""Email agent experiment — runs the agent on the trajectory dataset and scores it."""
import uuid
from typing import Any, List

from utils.config import client
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


def _run_email_final_response(inputs: dict) -> dict:
    """Run the assistant and return only the final text response.

    The completeness and professionalism LLM judges receive this output.
    Returning plain text (not the full messages list) avoids tool_call
    content blocks that newer OpenAI API versions reject in LLM judge calls.
    """
    result = email_assistant.invoke(inputs, config={"thread_id": uuid.uuid4()})
    for msg in reversed(result["messages"]):
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
        tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", [])
        msg_type = msg.get("type") if isinstance(msg, dict) else getattr(msg, "type", "")
        if msg_type == "ai" and content and not tool_calls:
            return {"output": content}
    return {"output": ""}


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

def load_experiments() -> None:
    print("Loading experiments...")
    print("     - Running trajectory experiment...")
    client.evaluate(
        _run_email_assistant,
        data="Email Agent: Trajectory",
        evaluators=[_evaluate_extra_steps],
        experiment_prefix="email-agent-trajectory",
        num_repetitions=1,
        max_concurrency=4,
    )
    print("     - Running final response experiment...")
    client.evaluate(
        _run_email_final_response,
        data="Email Agent: Final Response",
        evaluators=[],  # completeness + professionalism are rules-based evaluators on this dataset
        experiment_prefix="email-agent-final-response",
        num_repetitions=1,
        max_concurrency=4,
    )
    print("Experiments loaded successfully.")


if __name__ == "__main__":
    load_experiments()
