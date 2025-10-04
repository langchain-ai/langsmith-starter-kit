import uuid
from typing import Any, List

from setup.config import client
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

# Run Experiment
def run_trajectory_experiment() -> dict:   
    trajectory_dataset = "Email Agent: Trajectory"
    results = client.evaluate(
        run_email_assistant,
        data=trajectory_dataset,
        evaluators=[evaluate_extra_steps],
        experiment_prefix="email-agent-gpt4.1",
        num_repetitions=1,
        max_concurrency=4,
    )
    return results


def load_experiments() -> None:
    print("Loading experiments...")
    print("     - Running trajectory experiment...")
    run_trajectory_experiment()
    print("Experiments loaded successfully.")

if __name__ == "__main__":
    load_experiments()