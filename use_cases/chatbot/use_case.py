from use_cases.base import UseCase
from use_cases.chatbot.setup.datasets import load_datasets
from use_cases.chatbot.setup.evaluators import load_evaluators
from use_cases.chatbot.setup.experiments import load_experiments
from use_cases.chatbot.setup.traces import create_traces as _create_traces


class ChatbotUseCase(UseCase):
    name = "chatbot"
    project_name = "starter-kit-chatbot"
    tags = ["starter-kit", "use-case:chatbot"]

    def setup_prompts(self):
        pass  # Chatbot prompt is inline in agent.py

    def setup_datasets(self):
        load_datasets(self.use_api)

    def setup_evaluators(self):
        load_evaluators(self.use_api)

    def setup_experiments(self):
        load_experiments(self.use_api)

    def create_traces(self):
        _create_traces()
