"""Email agent evaluators — attaches scoring rules to email agent datasets."""
import os

from utils.evaluators import create_evaluator


def load_evaluators() -> None:
    print("Creating evaluators...")

    # --- LLM judge evaluators (hub prompts) ---
    create_evaluator(
        "correctness", "Email Agent: Next Action",
        prompt_or_ref="email-agent-next-action-eval:latest",
        score_type="boolean",
    )
    create_evaluator(
        "completeness", "Email Agent: Final Response",
        prompt_or_ref="email-agent-final-response-eval:latest",
        score_type="boolean",
    )
    create_evaluator(
        "professionalism", os.getenv("LANGSMITH_PROJECT"),
        target_type="project",
        prompt_or_ref="email-agent-professionalism-eval:latest",
        score_type="boolean",
    )

    # --- Code evaluators ---
    def perform_eval(run, example):
        correctness = (
            example["outputs"]["classification"].lower()
            == run["outputs"]["output"]["content"].split("\n\n")[0].lower()
        )
        return {"correctness": correctness}

    create_evaluator("triage_match", "Email Agent: Triage", func=perform_eval)

    def perform_eval(run, example):
        """Evaluate whether the trajectory exactly matches the expected output."""
        return {"exact_match": run["outputs"]["trajectory"] == example["outputs"]["trajectory"]}

    create_evaluator("trajectory_match", "Email Agent: Trajectory", func=perform_eval)

    print("Evaluators created.")


if __name__ == "__main__":
    load_evaluators()
