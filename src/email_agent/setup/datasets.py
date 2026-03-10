"""Email agent dataset setup — loads email examples into LangSmith datasets."""
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

from utils.datasets import create_langsmith_dataset

_DATA_DIR = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_emails() -> List[Dict]:
    rows = []
    with open(_DATA_DIR / "traces" / "ground_truth_emails.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["expected_tool_calls"] = (
                row["expected_tool_calls"].split("|") if row["expected_tool_calls"] else []
            )
            rows.append(row)
    return rows


def _load_next_action() -> Tuple[List[Dict], List[Dict]]:
    inputs, outputs = [], []
    with open(_DATA_DIR / "eval" / "ground_truth_next_action.jsonl", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            inputs.append(obj["input"])
            outputs.append(obj["output"])
    return inputs, outputs


# ---------------------------------------------------------------------------
# Dataset loaders — each is idempotent (skips if dataset already exists)
# ---------------------------------------------------------------------------

def load_datasets() -> None:
    print("Loading Datasets...")
    rows = _load_emails()

    email_inputs = [
        {"author": r["author"], "to": r["to"], "subject": r["subject"], "email_thread": r["email_thread"]}
        for r in rows
    ]

    print("   - Email Agent: Triage...")
    create_langsmith_dataset(
        "Email Agent: Triage",
        inputs=[{"email_input": e} for e in email_inputs],
        outputs=[{"classification": r["triage_output"]} for r in rows],
    )

    print("   - Email Agent: Final Response...")
    create_langsmith_dataset(
        "Email Agent: Final Response",
        inputs=[{"email_input": e} for e in email_inputs],
        outputs=[{"response_criteria": r["response_criteria"]} for r in rows],
    )

    print("   - Email Agent: Trajectory...")
    create_langsmith_dataset(
        "Email Agent: Trajectory",
        inputs=[{"email_input": e} for e in email_inputs],
        outputs=[{"trajectory": r["expected_tool_calls"]} for r in rows],
    )

    print("   - Email Agent: Next Action...")
    next_action_inputs, next_action_outputs = _load_next_action()
    create_langsmith_dataset(
        "Email Agent: Next Action",
        inputs=next_action_inputs,
        outputs=next_action_outputs,
    )


if __name__ == "__main__":
    load_datasets()
