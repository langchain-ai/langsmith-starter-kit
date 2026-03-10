"""Finance QA dataset setup — loads evaluation CSVs into LangSmith datasets."""
import csv
from pathlib import Path
from typing import Dict, List

from utils.datasets import create_langsmith_dataset

_DATA_DIR = Path(__file__).parent.parent / "data"


def _load_csv(filename: str) -> List[Dict]:
    with open(_DATA_DIR / filename, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _generate_reference_answer(question: str) -> str:
    """Generate a specific per-question reference answer using the KB and LLM."""
    from src.chatbot.agent.tools import search_knowledge_base
    from src.model import model

    results = search_knowledge_base(question, top_k=3, min_similarity=0.05)
    context = "\n\n".join(
        f"Topic: {r['question']}\nAnswer: {r['answer']}\nDetails: {r['retrieved_chunks'][:600]}"
        for r in results
    ) if results else "No relevant context found."

    response = model.invoke([
        {"role": "system", "content": (
            "You are a financial customer service expert. "
            "Answer the customer's question in 2-4 specific sentences based only on the provided knowledge base context. "
            "Be direct, specific, and accurate. Do not include generic preamble."
        )},
        {"role": "user", "content": f"Knowledge Base Context:\n{context}\n\nCustomer Question: {question}"},
    ])
    return response.content


def load_datasets() -> None:
    print("Loading Datasets...")

    print("   - Finance QA: Final Response...")
    rows = _load_csv("eval_dataset_ground_truth.csv")
    questions = [r["question"] for r in rows]
    print(f"     Generating {len(questions)} reference answers...")
    answers = [_generate_reference_answer(q) for q in questions]
    create_langsmith_dataset(
        "Finance QA: Final Response",
        inputs=[{"question": q} for q in questions],
        outputs=[{"answer": a} for a in answers],
    )

    print("   - Finance QA: RAG Citation...")
    rows = _load_csv("eval_dataset_multi_source.csv")
    create_langsmith_dataset(
        "Finance QA: RAG Citation",
        inputs=[{"question": r["question"]} for r in rows],
        outputs=[{"answer": r["answer"], "retrieved_chunks": r.get("retrieved_chunks", "")} for r in rows],
    )


if __name__ == "__main__":
    load_datasets()
