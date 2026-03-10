from typing import Optional

from src.base import UseCase
from src.chatbot.setup.datasets import load_datasets
from src.chatbot.setup.evaluators import load_evaluators
from src.chatbot.setup.experiments import load_experiments
from src.chatbot.setup.traces import create_traces as _create_traces


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

    def create_traces(self, num_traces: Optional[int] = None):
        _create_traces(num_traces=num_traces)
