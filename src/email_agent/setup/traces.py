"""Email agent trace generation — runs the agent on sampled emails to create traces."""
import csv
import random
from pathlib import Path
from typing import Optional

from src.email_agent.agent.agent import email_assistant

_DATA_DIR = Path(__file__).parent.parent / "data"


def _load_emails() -> list:
    with open(_DATA_DIR / "traces" / "ground_truth_emails.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def create_traces(num_traces: Optional[int] = None) -> None:
    """Run the email assistant on a random sample of emails to populate LangSmith traces.

    Args:
        num_traces: Number of emails to run. Defaults to all 16 emails.
    """
    print("Creating traces...")
    rows = _load_emails()
    sample = random.sample(rows, min(num_traces or len(rows), len(rows)))

    for row in sample:
        email_input = {
            "author": row["author"],
            "to": row["to"],
            "subject": row["subject"],
            "email_thread": row["email_thread"],
        }
        print(f"    - Running: {row['name']} — {row['subject']}")
        email_assistant.invoke(
            {"email_input": email_input},
            config={
                "run_name": f"email_assistant:{row['name']}",
                "tags": ["email-assistant", "trace", row["name"]],
            },
        )

    print(f"    - Generated {len(sample)} traces in LangSmith.")


if __name__ == "__main__":
    create_traces()
