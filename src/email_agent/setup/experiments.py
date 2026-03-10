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
        if getattr(msg, "tool_calls", None):
            names.extend(call["name"].lower() for call in getattr(msg, "tool_calls", None))
    return names


def _run_email_assistant(inputs: dict) -> dict:
    result = email_assistant.invoke(inputs, config={"thread_id": uuid.uuid4()})
    return {"trajectory": _extract_tool_calls(result["messages"])}


def _serialize_messages(messages: list) -> list:
    """Convert LangGraph messages to a JSON-serializable list of dicts."""
    result = []
    for msg in messages:
        entry = {"role": msg.type}
        if msg.content:
            entry["content"] = msg.content
        if getattr(msg, "tool_calls", None):
            entry["tool_calls"] = [
                {"name": c["name"], "args": c["args"]}
                for c in getattr(msg, "tool_calls", None)
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
        for call in (getattr(msg, "tool_calls", None) or []):
            if call["name"] == "write_email":
                args = call["args"]
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
