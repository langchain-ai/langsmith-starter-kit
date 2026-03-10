"""Chatbot trace generation — runs the chatbot on sampled questions to create traces."""
import asyncio
import csv
import random
import sys
from pathlib import Path
from typing import Optional

from src.chatbot.agent.agent import chatbot

_DATA_DIR = Path(__file__).parent.parent / "data"


def _load_question_bank() -> list:
    path = _DATA_DIR / "question_bank.csv"
    if not path.exists():
        raise FileNotFoundError(f"Question bank not found at {path}")
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


async def _run_chatbot_async(row: dict) -> dict:
    try:
        result = await asyncio.to_thread(
            chatbot.invoke,
            {"messages": [{"role": "user", "content": row["question"]}]},
        )
        messages = result.get("messages", [])
        answer = None
        for msg in reversed(messages):
            content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
            tool_calls = getattr(msg, "tool_calls", []) or (msg.get("tool_calls") if isinstance(msg, dict) else [])
            msg_type = getattr(msg, "type", None) or (msg.get("type") if isinstance(msg, dict) else None)
            if msg_type == "ai" and content and not tool_calls:
                answer = content
                break
        return {**row, "answer": answer or "No response", "success": True}
    except Exception as e:
        return {**row, "answer": None, "success": False, "error": str(e)}


async def _generate_traces_async(questions: list, max_concurrent: int) -> None:
    total = len(questions)
    print(f"Running {total} questions through chatbot (max concurrency: {max_concurrent})...")
    for i in range(0, total, max_concurrent):
        batch = questions[i:i + max_concurrent]
        batch_num = i // max_concurrent + 1
        total_batches = (total + max_concurrent - 1) // max_concurrent
        print(f"  Batch {batch_num}/{total_batches}")
        sys.stdout.flush()
        results = await asyncio.gather(*[_run_chatbot_async(q) for q in batch])
        for j, r in enumerate(results):
            status = "✓" if r["success"] else "✗"
            print(f"  {status} [{i + j + 1:2d}/{total}] {r['question'][:60]}...")
        sys.stdout.flush()


def create_traces(num_traces: Optional[int] = None, max_concurrent: int = 3) -> None:
    """Generate traces by running the chatbot on questions from the question bank.

    Args:
        num_traces: Number of traces to generate. Defaults to 20.
        max_concurrent: Maximum concurrent chatbot invocations.
    """
    print("Creating traces...")
    bank = _load_question_bank()
    n = min(num_traces or 20, len(bank))
    questions = random.sample(bank, n)
    asyncio.run(_generate_traces_async(questions, max_concurrent))
    print(f"    - Generated {n} traces in LangSmith.")


if __name__ == "__main__":
    create_traces()
