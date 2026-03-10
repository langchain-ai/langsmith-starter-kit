from setup.config import client
from use_cases.chatbot.agent.agent import chatbot


def run_chatbot(inputs: dict) -> dict:
    """Run the chatbot on a question and return messages."""
    question = inputs.get("question", "")
    result = chatbot.invoke({"messages": [{"role": "user", "content": question}]})
    return {"messages": result.get("messages", [])}


def evaluate_has_response(outputs: dict, reference_outputs: dict) -> dict:
    """Check that the chatbot produced a non-empty final AI response."""
    messages = outputs.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "type"):
            if msg.type == "ai" and getattr(msg, "content", "") and not getattr(msg, "tool_calls", []):
                return {"key": "has_response", "score": 1}
        elif isinstance(msg, dict):
            if msg.get("type") == "ai" and msg.get("content") and not msg.get("tool_calls"):
                return {"key": "has_response", "score": 1}
    return {"key": "has_response", "score": 0}


def load_experiments(use_api: bool = False) -> None:
    print("Loading experiments...")
    print("     - Running chatbot evaluation experiment...")
    client.evaluate(
        run_chatbot,
        data="Chatbot: Ground Truth",
        evaluators=[evaluate_has_response],
        experiment_prefix="chatbot-gpt4.1",
        num_repetitions=1,
        max_concurrency=3,
    )
    print("Experiments loaded successfully.")


if __name__ == "__main__":
    load_experiments()
