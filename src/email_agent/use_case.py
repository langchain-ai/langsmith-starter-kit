from typing import Optional

from src.base import UseCase
from src.email_agent.setup.prompts import load_all_prompts
from src.email_agent.setup.datasets import load_datasets
from src.email_agent.setup.evaluators import load_evaluators
from src.email_agent.setup.experiments import load_experiments
from src.email_agent.setup.annotations import load_automations_and_queues
from src.email_agent.setup.traces import create_traces as _create_traces


class EmailAgentUseCase(UseCase):
    name = "email-agent"
    project_name = "starter-email-agent"
    tags = ["starter-kit", "starter:email-agent"]
    dataset_names = [
        "Email Agent: Triage",
        "Email Agent: Final Response",
        "Email Agent: Trajectory",
        "Email Agent: Next Action",
    ]
    prompt_names = [
        "email-agent-action",
        "email-agent-triage",
        "email-agent-next-action-eval",
        "email-agent-final-response-eval",
        "email-agent-professionalism-eval",
        "guardrail-example",
    ]
    queue_names = ["Professionalism Annotation Queue"]

    def setup_prompts(self):
        load_all_prompts()

    def setup_datasets(self):
        load_datasets()

    def setup_evaluators(self):
        load_evaluators()

    def setup_experiments(self):
        load_experiments()

    def setup_annotations(self):
        load_automations_and_queues()

    def create_traces(self, num_traces: Optional[int] = None):
        _create_traces(num_traces=num_traces)
