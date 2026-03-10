import asyncio
import csv
import random
import sys
from pathlib import Path
from typing import List, Dict

from use_cases.chatbot.agent.agent import chatbot


def _load_question_bank() -> List[Dict[str, str]]:
    """Load questions from the question bank CSV."""
    bank_path = Path(__file__).parent.parent / "data" / "question_bank.csv"
    if not bank_path.exists():
        raise FileNotFoundError(f"Question bank not found at {bank_path}")

    questions = []
    with open(bank_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append({
                "id": row["id"],
                "question": row["question"],
                "in_scope": row.get("in_scope", "yes") == "yes",
                "category": row.get("category", ""),
            })
    return questions


async def _run_chatbot_async(question_data: Dict) -> Dict:
    """Run the chatbot on a single question asynchronously."""
    try:
        question = question_data["question"]
        input_data = {"messages": [{"role": "user", "content": question}]}
        result = await asyncio.to_thread(chatbot.invoke, input_data)

        messages = result.get("messages", [])
        final_message = None
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "ai" and hasattr(msg, "content"):
                if msg.content and not getattr(msg, "tool_calls", []):
                    final_message = msg.content
                    break

        return {**question_data, "answer": final_message or "No response", "success": True}
    except Exception as e:
        return {**question_data, "answer": None, "success": False, "error": str(e)}


async def _generate_traces_async(questions: List[Dict], max_concurrent: int = 3):
    """Generate traces by running questions through the chatbot."""
    print(f"Running {len(questions)} questions through chatbot (max concurrency: {max_concurrent})...")
    results = []

    for i in range(0, len(questions), max_concurrent):
        batch = questions[i:i + max_concurrent]
        batch_num = (i // max_concurrent) + 1
        total_batches = (len(questions) + max_concurrent - 1) // max_concurrent

        print(f"  Batch {batch_num}/{total_batches}")
        sys.stdout.flush()

        tasks = [_run_chatbot_async(q) for q in batch]
        batch_results = await asyncio.gather(*tasks)

        for j, result in enumerate(batch_results):
            q_num = i + j + 1
            status = "✓" if result["success"] else "✗"
            print(f"  {status} [{q_num:2d}/{len(questions)}] {result['question'][:60]}...")
            sys.stdout.flush()

        results.extend(batch_results)

    return results


def create_traces(num_traces: int = 20, max_concurrent: int = 3) -> None:
    """Generate traces by running chatbot on questions from the question bank.

    Args:
        num_traces: Number of traces to generate (default: 20)
        max_concurrent: Maximum concurrent chatbot invocations (default: 3)
    """
    print("Creating traces...")
    question_bank = _load_question_bank()

    # Select questions (cap at bank size)
    num_to_run = min(num_traces, len(question_bank))
    questions = random.sample(question_bank, num_to_run)

    asyncio.run(_generate_traces_async(questions, max_concurrent=max_concurrent))
    print(f"    - Generated {num_to_run} traces in LangSmith.")


if __name__ == "__main__":
    create_traces()
