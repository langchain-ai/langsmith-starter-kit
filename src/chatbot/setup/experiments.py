"""Chatbot experiment — runs the chatbot on the Ground Truth dataset."""
from utils.config import client
from src.chatbot.agent.agent import chatbot


def _run_chatbot(inputs: dict) -> dict:
    question = inputs.get("question", "")
    result = chatbot.invoke({"messages": [{"role": "user", "content": question}]})
    return {"messages": result.get("messages", [])}


def _evaluate_has_response(outputs: dict, reference_outputs: dict) -> dict:
    """Check that the chatbot produced a non-empty final AI response."""
    for msg in reversed(outputs.get("messages", [])):
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
        tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", [])
        msg_type = msg.get("type") if isinstance(msg, dict) else getattr(msg, "type", "")
        if msg_type == "ai" and content and not tool_calls:
            return {"key": "has_response", "score": 1}
    return {"key": "has_response", "score": 0}


def load_experiments(use_api: bool = False) -> None:
    print("Loading experiments...")
    print("     - Running chatbot evaluation experiment...")
    client.evaluate(
        _run_chatbot,
        data="Chatbot: Ground Truth",
        evaluators=[_evaluate_has_response],
        experiment_prefix="chatbot-gpt4.1",
        num_repetitions=1,
        max_concurrency=3,
    )
    print("Experiments loaded successfully.")


if __name__ == "__main__":
    load_experiments()
