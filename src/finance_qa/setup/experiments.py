"""Finance QA experiments — runs the chatbot on evaluation datasets and scores results."""
from langchain_core.messages import HumanMessage
from utils.config import client
from src.finance_qa.agent.agent import chatbot


def _run_chatbot_final_response(inputs: dict) -> dict:
    """Run the chatbot and return only the final text response.

    The helpfulness and answer_correctness LLM judges receive this output.
    Returning plain text avoids tool_call content blocks that OpenAI rejects.
    """
    question = inputs.get("question", "")
    result = chatbot.invoke({"messages": [HumanMessage(content=question)]})
    for msg in reversed(result.get("messages", [])):
        if msg.type == "ai" and msg.content and not msg.tool_calls:
            return {"output": msg.content}
    return {"output": ""}


def _run_chatbot(inputs: dict) -> dict:
    """Run the chatbot and return full messages for citation code evaluators.

    Also returns 'output' (final text) so LLM judge evaluators receive plain text
    instead of tool_call content blocks, which OpenAI Chat Completions rejects.
    """
    question = inputs.get("question", "")
    result = chatbot.invoke({"messages": [HumanMessage(content=question)]})
    messages = result.get("messages", [])
    final_text = ""
    for msg in reversed(messages):
        if msg.type == "ai" and msg.content and not msg.tool_calls:
            final_text = msg.content
            break
    return {"messages": messages, "output": final_text}


def _evaluate_has_response(outputs: dict, reference_outputs: dict) -> dict:
    """Check that the chatbot produced a non-empty final AI response."""
    for msg in reversed(outputs.get("messages", [])):
        if msg.type == "ai" and msg.content and not msg.tool_calls:
            return {"key": "has_response", "score": 1}
    return {"key": "has_response", "score": 0}


def load_experiments() -> None:
    print("Loading experiments...")
    print("     - Running final response experiment...")
    client.evaluate(
        _run_chatbot_final_response,
        data="Finance QA: Final Response",
        evaluators=[],  # helpfulness + answer_correctness are online evaluator rules
        experiment_prefix="finance-qa-final-response",
        num_repetitions=1,
        max_concurrency=3,
    )
    print("     - Running RAG citation experiment...")
    client.evaluate(
        _run_chatbot,
        data="Finance QA: RAG Citation",
        evaluators=[_evaluate_has_response],
        experiment_prefix="finance-qa-rag-citation",
        num_repetitions=1,
        max_concurrency=3,
    )
    print("Experiments loaded successfully.")


if __name__ == "__main__":
    load_experiments()
