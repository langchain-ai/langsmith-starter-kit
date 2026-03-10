"""Chatbot evaluators — citation and correctness scoring rules."""
from typing import Optional

from utils.evaluators import create_evaluator


_CITATION_ACCURACY_PROMPT = [
    ["system",
     "You are an expert evaluator assessing citation accuracy in customer service responses. "
     "Given a question, an agent's response (which may include tool search results and a final answer), "
     "and a reference answer, rate how accurately the agent's citations support the claims made. "
     "Score 1.0 if all claims are accurately cited and consistent with the reference; 0.0 if no citations or completely wrong."],
    ["human",
     "Question: {input}\n\nAgent Response: {output}\n\nReference Answer: {reference}\n\n"
     "Provide a citation_accuracy score from 0.0 to 1.0."],
]

_ANSWER_CORRECTNESS_PROMPT = [
    ["system",
     "You are an expert evaluator assessing the correctness of customer service responses. "
     "Given a question, an agent's response, and a reference answer, rate how correct and complete "
     "the agent's answer is. Score 1.0 if the answer is fully correct and consistent with the reference; "
     "0.0 if it is completely wrong or missing the key information."],
    ["human",
     "Question: {input}\n\nAgent Response: {output}\n\nReference Answer: {reference}\n\n"
     "Provide an answer_correctness score from 0.0 to 1.0."],
]


def load_evaluators(use_api: bool = False, owner: Optional[str] = None) -> None:
    print("Creating evaluators...")

    # Code evaluators on the Multi-Source dataset
    def perform_eval(run, example):
        import re
        messages = run.get("outputs", {}).get("messages", [])
        response = ""
        for msg in reversed(messages):
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
            tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", [])
            msg_type = msg.get("type") if isinstance(msg, dict) else getattr(msg, "type", "")
            if msg_type == "ai" and content and not tool_calls:
                response = content
                break
        if not response:
            return {"citation_presence": False}
        match = re.search(r'Relevant docs?:\s*(.*?)(?:\n\n|\Z)', response, re.IGNORECASE | re.DOTALL)
        if not match:
            return {"citation_presence": False}
        sources = [l.strip() for l in match.group(1).split('\n') if l.strip().startswith('-')]
        return {"citation_presence": len(sources) > 0}

    create_evaluator("citation_presence", "Chatbot: Multi-Source", func=perform_eval)

    def perform_eval(run, example):
        import re
        messages = run.get("outputs", {}).get("messages", [])
        response = ""
        for msg in reversed(messages):
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
            tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", [])
            msg_type = msg.get("type") if isinstance(msg, dict) else getattr(msg, "type", "")
            if msg_type == "ai" and content and not tool_calls:
                response = content
                break
        if not response:
            return {"citation_grounding": 0.0}
        match = re.search(r'Relevant docs?:\s*(.*?)(?:\n\n|\Z)', response, re.IGNORECASE | re.DOTALL)
        if not match:
            return {"citation_grounding": 0.0}
        sources = []
        for line in match.group(1).split('\n'):
            line = line.strip()
            if line.startswith('-'):
                source = re.sub(r'^\s*-\s*', '', line).strip()
                source = re.sub(r'\[([^\]]+)\]\([^\)]*\)', r'\1', source)
                source = re.sub(r'\s*\(.*?\)\s*', '', source).strip()
                if source:
                    sources.append(source.lower())
        if not sources:
            return {"citation_grounding": 0.0}
        gt_chunks = example.get("outputs", {}).get("retrieved_chunks", "").lower()
        grounded = sum(1 for s in sources if s in gt_chunks)
        return {"citation_grounding": grounded / len(sources)}

    create_evaluator("citation_grounding", "Chatbot: Multi-Source", func=perform_eval)

    # LLM judge evaluators on the Ground Truth dataset
    create_evaluator(
        "citation_accuracy", "Chatbot: Ground Truth",
        prompt_or_ref=_CITATION_ACCURACY_PROMPT, score_type="number",
    )
    create_evaluator(
        "answer_correctness", "Chatbot: Ground Truth",
        prompt_or_ref=_ANSWER_CORRECTNESS_PROMPT, score_type="number",
    )

    print("Evaluators created.")


if __name__ == "__main__":
    load_evaluators()
