"""Finance QA trace generation — runs the chatbot on sampled questions to create traces."""
import asyncio
import csv
import math
import random
import sys
from pathlib import Path
from typing import Dict, Optional

from langchain_core.messages import HumanMessage

from src.finance_qa.agent.agent import chatbot

_DATA_DIR = Path(__file__).parent.parent / "data"


def _load_question_bank() -> list:
    path = _DATA_DIR / "traces" / "synthetic_question_bank.csv"
    if not path.exists():
        raise FileNotFoundError(f"Question bank not found at {path}")
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _sample_with_distribution(
    bank: list,
    n: int,
    distribution: Dict[str, float],
    column: str = "relevance",
) -> list:
    """Sample n rows from bank proportionally by distribution.

    Args:
        bank: Full question bank rows.
        n: Total number of rows to return.
        distribution: Mapping of column value → desired ratio. Keys must match
            values found in ``column`` (e.g. ``{"in_scope": 0.4, ...}``).
        column: Row field to group by (default: ``"relevance"``).
    """
    by_value = {}
    for key in distribution:
        by_value[key] = [r for r in bank if r.get(column) == key]

    selected = []
    for key, ratio in distribution.items():
        count = math.floor(n * ratio)
        pool = by_value[key]
        selected.extend(random.sample(pool, min(count, len(pool))))

    shortfall = n - len(selected)
    if shortfall > 0:
        already = set(id(r) for r in selected)
        remaining = [r for r in bank if id(r) not in already]
        selected.extend(random.sample(remaining, min(shortfall, len(remaining))))

    random.shuffle(selected)
    return selected


async def _run_chatbot_async(row: dict) -> dict:
    try:
        result = await asyncio.to_thread(
            chatbot.invoke,
            {"messages": [HumanMessage(content=row["question"])]},
        )
        answer = None
        for msg in reversed(result.get("messages", [])):
            if msg.type == "ai" and msg.content and not msg.tool_calls:
                answer = msg.content
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


def create_traces(
    num_traces: Optional[int] = None,
    distribution: Optional[Dict[str, float]] = None,
    distribution_column: str = "relevance",
    max_concurrent: int = 3,
) -> None:
    """Generate traces by running the chatbot on questions from the question bank.

    Args:
        num_traces: Number of traces to generate. Defaults to 20.
        distribution: Mapping of column value → ratio for stratified sampling.
            Keys must match values in ``distribution_column``. When None, samples randomly.
        distribution_column: Question bank column to stratify by (default: ``"relevance"``).
        max_concurrent: Maximum concurrent chatbot invocations.
    """
    print("Creating traces...")
    bank = _load_question_bank()
    n = min(num_traces or 20, len(bank))

    if distribution is not None:
        questions = _sample_with_distribution(bank, n, distribution, column=distribution_column)
    else:
        questions = random.sample(bank, n)

    asyncio.run(_generate_traces_async(questions, max_concurrent))
    print(f"    - Generated {n} traces in LangSmith.")


if __name__ == "__main__":
    create_traces()
