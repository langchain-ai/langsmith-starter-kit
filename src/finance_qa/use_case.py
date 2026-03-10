from typing import Optional

from src.base import UseCase
from src.finance_qa.setup.prompts import load_all_prompts
from src.finance_qa.setup.datasets import load_datasets
from src.finance_qa.setup.evaluators import load_evaluators
from src.finance_qa.setup.experiments import load_experiments
from src.finance_qa.setup.annotations import load_automations_and_queues
from src.finance_qa.setup.traces import create_traces as _create_traces


class FinanceQAUseCase(UseCase):
    name = "finance-qa"
    project_name = "starter-finance-qa"
    tags = ["starter-kit", "starter:finance-qa"]
    dataset_names = [
        "Finance QA: Final Response",
        "Finance QA: RAG Citation",
    ]
    prompt_names = [
        "finance-qa-helpfulness-eval",
        "finance-qa-helpfulness-online-eval",
        "finance-qa-rag-citation-eval",
        "finance-qa-answer-correctness-eval",
    ]
    queue_names = ["Finance QA: Helpfulness Review Queue"]

    default_distribution = {
        "in_scope": 0.4,
        "irrelevant_match": 0.3,
        "out_of_scope": 0.3,
    }

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
        _create_traces(
            num_traces=num_traces,
            distribution=self.default_distribution,
        )
