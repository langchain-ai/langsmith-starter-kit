from use_cases.base import UseCase
from use_cases.email_agent.setup.prompts import load_all_prompts
from use_cases.email_agent.setup.datasets import load_datasets
from use_cases.email_agent.setup.evaluators import load_evaluators
from use_cases.email_agent.setup.experiments import load_experiments
from use_cases.email_agent.setup.annotations import load_automations_and_queues
from use_cases.email_agent.setup.traces import create_traces as _create_traces


class EmailAgentUseCase(UseCase):
    name = "email-agent"
    project_name = "starter-kit-email-agent"
    tags = ["starter-kit", "use-case:email-agent"]

    def setup_prompts(self):
        load_all_prompts(self.use_api)

    def setup_datasets(self):
        load_datasets(self.use_api)

    def setup_evaluators(self):
        load_evaluators(self.use_api)

    def setup_experiments(self):
        load_experiments(self.use_api)

    def setup_annotations(self):
        load_automations_and_queues()

    def create_traces(self):
        _create_traces()
