"""Finance QA evaluators — citation, helpfulness, and correctness scoring rules."""
import os

from utils.evaluators import create_evaluator


def load_evaluators() -> None:
    print("Creating evaluators...")

    # --- Code evaluators on RAG Citation dataset ---
    # NOTE: These functions are serialized and executed server-side in a sandboxed
    # Pyodide environment, so all imports must be inside the function body.
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

    create_evaluator("citation_presence", "Finance QA: RAG Citation", func=perform_eval)

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

    create_evaluator("citation_grounding", "Finance QA: RAG Citation", func=perform_eval)

    # --- LLM judge evaluators on RAG Citation dataset ---
    create_evaluator(
        "rag_citation_quality", "Finance QA: RAG Citation",
        prompt_or_ref="finance-qa-rag-citation-eval:latest", score_type="number",
    )

    # --- LLM judge evaluators on Final Response dataset ---
    create_evaluator(
        "helpfulness", "Finance QA: Final Response",
        prompt_or_ref="finance-qa-helpfulness-eval:latest", score_type="boolean",
    )
    create_evaluator(
        "answer_correctness", "Finance QA: Final Response",
        prompt_or_ref="finance-qa-answer-correctness-eval:latest", score_type="number",
    )

    # --- Project-level helpfulness evaluator (no reference available) ---
    create_evaluator(
        "helpfulness", os.getenv("LANGSMITH_PROJECT"),
        target_type="project",
        prompt_or_ref="finance-qa-helpfulness-online-eval:latest", score_type="boolean",
    )

    print("Evaluators created.")


if __name__ == "__main__":
    load_evaluators()
