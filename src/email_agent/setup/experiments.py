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


def _serialize_messages(messages: list) -> list:
    """Convert LangGraph messages to a JSON-serializable list of dicts."""
    result = []
    for msg in messages:
        msg_type = msg.get("type") if isinstance(msg, dict) else getattr(msg, "type", "")
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
        tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", [])
        entry = {"role": msg_type}
        if content:
            entry["content"] = content
        if tool_calls:
            entry["tool_calls"] = [
                {
                    "name": c.get("name") if isinstance(c, dict) else getattr(c, "name", ""),
                    "args": c.get("args") if isinstance(c, dict) else getattr(c, "args", {}),
                }
                for c in tool_calls
            ]
        result.append(entry)
    return result


def _run_email_final_response(inputs: dict) -> dict:
    """Run the assistant and return the full LangGraph state as output.

    The LLM judges (completeness, professionalism) receive the full tool call
    history via variable_mapping, so they can evaluate both the classification
    decision and the quality of the written email. The email text is also
    surfaced as a separate 'email' key for convenience.
    """
    result = email_assistant.invoke(inputs, config={"thread_id": uuid.uuid4()})
    classification = result.get("classification_decision", "")
    messages = result.get("messages", [])

    email = ""
    for msg in messages:
        tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", [])
        for call in (tool_calls or []):
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
            if name == "write_email":
                args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
                email = f"Subject: {args.get('subject', '')}\n\n{args.get('content', '')}"

    return {
        "output": {
            "classification": classification,
            "messages": _serialize_messages(messages),
        },
        "email": email,
    }


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
