"""Finance QA trace generation — runs the chatbot on sampled questions to create traces."""
import asyncio
import csv
import math
import random
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage

from src.finance_qa.agent.agent import chatbot

_DATA_DIR = Path(__file__).parent.parent / "data"

# Multi-turn conversation templates.  Each is a list of user messages that
# form a natural follow-up conversation on a single topic.
MULTI_TURN_CONVERSATIONS: List[List[str]] = [
    [
        "How does your rewards program work?",
        "Can I transfer my points to an airline loyalty program?",
        "What happens to my points if I close my card?",
    ],
    [
        "I just received my new credit card in the mail. How do I activate it?",
        "Great, it's activated. Can I set up contactless payments with it?",
        "What's my credit limit on this new card?",
    ],
    [
        "I noticed a charge on my statement that I don't recognize. How do I dispute it?",
        "How long does the dispute process usually take?",
        "Will I get a temporary credit while the investigation is ongoing?",
    ],
    [
        "I'd like to do a balance transfer from my other credit card. How does that work?",
        "Is there a fee for the balance transfer?",
        "How long does it take for the transfer to go through?",
    ],
    [
        "I'm worried about fraud on my account. What protections do you offer?",
        "How do I set up transaction alerts so I'm notified of suspicious activity?",
        "If my card is compromised, how quickly can I get a replacement?",
    ],
    [
        "Can you explain the interest charges on my latest statement?",
        "If I pay my full balance this month, will I still be charged interest next month?",
        "What's my current APR?",
    ],
    [
        "I want to close my credit card account. What steps do I need to take?",
        "I still have some rewards points remaining. Can I redeem them before closing?",
        "Will closing this card affect my credit score?",
    ],
    [
        "I lost my credit card. What should I do?",
        "Can I still use my card number for online purchases while I wait for the replacement?",
        "How long will it take to receive the new card?",
    ],
]


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
        distribution: Mapping of column value -> desired ratio. Keys must match
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
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    try:
        result = await asyncio.to_thread(
            chatbot.invoke,
            {"messages": [HumanMessage(content=row["question"])]},
            config,
        )
        answer = None
        for msg in reversed(result.get("messages", [])):
            if msg.type == "ai" and msg.content and not msg.tool_calls:
                answer = msg.content
                break
        return {**row, "answer": answer or "No response", "success": True}
    except Exception as e:
        return {**row, "answer": None, "success": False, "error": str(e)}


async def _run_multi_turn_async(conversation: List[str]) -> dict:
    """Run a multi-turn conversation on a single thread."""
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    turns_completed = 0
    try:
        for message in conversation:
            result = await asyncio.to_thread(
                chatbot.invoke,
                {"messages": [HumanMessage(content=message)]},
                config,
            )
            turns_completed += 1
        return {
            "conversation": conversation,
            "turns": turns_completed,
            "thread_id": thread_id,
            "success": True,
        }
    except Exception as e:
        return {
            "conversation": conversation,
            "turns": turns_completed,
            "thread_id": thread_id,
            "success": False,
            "error": str(e),
        }


async def _generate_traces_async(
    questions: list,
    conversations: List[List[str]],
    max_concurrent: int,
) -> None:
    # --- single-turn traces ---
    total = len(questions)
    if total:
        print(f"Running {total} single-turn questions (max concurrency: {max_concurrent})...")
        for i in range(0, total, max_concurrent):
            batch = questions[i : i + max_concurrent]
            batch_num = i // max_concurrent + 1
            total_batches = (total + max_concurrent - 1) // max_concurrent
            print(f"  Batch {batch_num}/{total_batches}")
            sys.stdout.flush()
            results = await asyncio.gather(*[_run_chatbot_async(q) for q in batch])
            for j, r in enumerate(results):
                status = "+" if r["success"] else "x"
                print(f"  {status} [{i + j + 1:2d}/{total}] {r['question'][:60]}...")
            sys.stdout.flush()

    # --- multi-turn traces ---
    if conversations:
        print(f"Running {len(conversations)} multi-turn conversations...")
        sys.stdout.flush()
        for idx, conv in enumerate(conversations, 1):
            r = await _run_multi_turn_async(conv)
            status = "+" if r["success"] else "x"
            print(f"  {status} [conversation {idx}/{len(conversations)}] "
                  f"{r['turns']} turns - {conv[0][:50]}...")
            sys.stdout.flush()


def create_traces(
    num_traces: Optional[int] = None,
    distribution: Optional[Dict[str, float]] = None,
    distribution_column: str = "relevance",
    max_concurrent: int = 3,
    num_conversations: Optional[int] = None,
) -> None:
    """Generate traces by running the chatbot on questions from the question bank.

    Args:
        num_traces: Number of single-turn traces to generate. Defaults to 20.
        distribution: Mapping of column value -> ratio for stratified sampling.
            Keys must match values in ``distribution_column``. When None, samples randomly.
        distribution_column: Question bank column to stratify by (default: ``"relevance"``).
        max_concurrent: Maximum concurrent chatbot invocations.
        num_conversations: Number of multi-turn conversations to generate.
            Defaults to 3. Set to 0 to skip.
    """
    print("Creating traces...")
    bank = _load_question_bank()
    n = min(num_traces or 20, len(bank))

    if distribution is not None:
        questions = _sample_with_distribution(bank, n, distribution, column=distribution_column)
    else:
        questions = random.sample(bank, n)

    n_conv = num_conversations if num_conversations is not None else 3
    conversations = random.sample(
        MULTI_TURN_CONVERSATIONS, min(n_conv, len(MULTI_TURN_CONVERSATIONS))
    )

    asyncio.run(_generate_traces_async(questions, conversations, max_concurrent))
    print(f"    - Generated {n} single-turn traces and {len(conversations)} multi-turn conversations in LangSmith.")


if __name__ == "__main__":
    create_traces()
