from typing import Optional

from use_cases.email_agent.setup.evaluators import (
    create_code_payload,
    create_judge_payload,
    create_evaluator,
)


# ------------------------------------------------------------------------------------------------------------------------
# CODE EVALUATORS (citation_presence, citation_grounding) — on "Chatbot: Multi-Source"
# ------------------------------------------------------------------------------------------------------------------------

def _citation_presence_eval():
    """Return a self-contained perform_eval function for citation presence."""
    def perform_eval(run, example):
        import re
        messages = run.get("outputs", {}).get("messages", [])
        response = ""
        for msg in reversed(messages):
            if isinstance(msg, dict):
                if msg.get("type") == "ai" and msg.get("content") and not msg.get("tool_calls"):
                    response = msg["content"]
                    break
            elif hasattr(msg, "type"):
                if msg.type == "ai" and getattr(msg, "content", "") and not getattr(msg, "tool_calls", []):
                    response = msg.content
                    break
        if not response:
            return {"citation_presence": False}
        docs_match = re.search(r'Relevant docs?:\s*(.*?)(?:\n\n|\Z)', response, re.IGNORECASE | re.DOTALL)
        if not docs_match:
            return {"citation_presence": False}
        sources = [l.strip() for l in docs_match.group(1).split('\n') if l.strip().startswith('-')]
        return {"citation_presence": len(sources) > 0}
    return perform_eval


def _citation_grounding_eval():
    """Return a self-contained perform_eval function for citation grounding."""
    def perform_eval(run, example):
        import re
        messages = run.get("outputs", {}).get("messages", [])
        response = ""
        for msg in reversed(messages):
            if isinstance(msg, dict):
                if msg.get("type") == "ai" and msg.get("content") and not msg.get("tool_calls"):
                    response = msg["content"]
                    break
            elif hasattr(msg, "type"):
                if msg.type == "ai" and getattr(msg, "content", "") and not getattr(msg, "tool_calls", []):
                    response = msg.content
                    break
        if not response:
            return {"citation_grounding": 0.0}
        docs_match = re.search(r'Relevant docs?:\s*(.*?)(?:\n\n|\Z)', response, re.IGNORECASE | re.DOTALL)
        if not docs_match:
            return {"citation_grounding": 0.0}
        sources = []
        for line in docs_match.group(1).split('\n'):
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
    return perform_eval


# ------------------------------------------------------------------------------------------------------------------------
# LLM JUDGE EVALUATORS (citation_accuracy, answer_correctness) — on "Chatbot: Ground Truth"
# ------------------------------------------------------------------------------------------------------------------------

_CITATION_ACCURACY_PROMPT = [
    [
        "system",
        "You are an expert evaluator assessing citation accuracy in customer service responses. "
        "Given a question, an agent's response (which may include tool search results and a final answer), "
        "and a reference answer, rate how accurately the agent's citations support the claims made. "
        "Score 1.0 if all claims are accurately cited and consistent with the reference; 0.0 if no citations or completely wrong."
    ],
    [
        "human",
        "Question: {input}\n\nAgent Response: {output}\n\nReference Answer: {reference}\n\n"
        "Provide a citation_accuracy score from 0.0 to 1.0."
    ]
]

_ANSWER_CORRECTNESS_PROMPT = [
    [
        "system",
        "You are an expert evaluator assessing the correctness of customer service responses. "
        "Given a question, an agent's response, and a reference answer, rate how correct and complete "
        "the agent's answer is. Score 1.0 if the answer is fully correct and consistent with the reference; "
        "0.0 if it is completely wrong or missing the key information."
    ],
    [
        "human",
        "Question: {input}\n\nAgent Response: {output}\n\nReference Answer: {reference}\n\n"
        "Provide an answer_correctness score from 0.0 to 1.0."
    ]
]


def load_evaluators(use_api: bool = False, owner: Optional[str] = None) -> None:
    """Create citation and correctness evaluators on chatbot datasets."""
    print("Creating evaluators...")

    multi_source_dataset = "Chatbot: Multi-Source"
    ground_truth_dataset = "Chatbot: Ground Truth"

    # 1. citation_presence (code) — Multi-Source
    citation_presence_payload = create_code_payload(
        name="citation_presence",
        func=_citation_presence_eval(),
        language="python",
        sample_rate=1.0,
        target_name=multi_source_dataset,
        target_type="dataset",
    )

    # 2. citation_grounding (code) — Multi-Source
    citation_grounding_payload = create_code_payload(
        name="citation_grounding",
        func=_citation_grounding_eval(),
        language="python",
        sample_rate=1.0,
        target_name=multi_source_dataset,
        target_type="dataset",
    )

    # 3. citation_accuracy (LLM judge, inline prompt) — Ground Truth
    citation_accuracy_payload = create_judge_payload(
        name="citation_accuracy",
        prompt_or_ref=_CITATION_ACCURACY_PROMPT,
        sample_rate=1.0,
        score_type="number",
        target_name=ground_truth_dataset,
        target_type="dataset",
        use_api=use_api,
        owner=owner,
    )

    # 4. answer_correctness (LLM judge, inline prompt) — Ground Truth
    answer_correctness_payload = create_judge_payload(
        name="answer_correctness",
        prompt_or_ref=_ANSWER_CORRECTNESS_PROMPT,
        sample_rate=1.0,
        score_type="number",
        target_name=ground_truth_dataset,
        target_type="dataset",
        use_api=use_api,
        owner=owner,
    )

    evaluators = [
        citation_presence_payload,
        citation_grounding_payload,
        citation_accuracy_payload,
        answer_correctness_payload,
    ]

    for payload in evaluators:
        try:
            if payload:
                create_evaluator(payload)
                print(f"    - Evaluator '{payload.display_name}' created.")
        except Exception as e:
            print(f"    - Error creating evaluator: {e}")
            continue


if __name__ == "__main__":
    load_evaluators()
