import os
import time
from abc import ABC, abstractmethod

from setup.config import setup_project, setup_secrets


class UseCase(ABC):
    name: str           # slug, e.g. "email-agent"
    project_name: str   # LangSmith project, e.g. "starter-kit-email-agent"
    tags: list          # Application tags, e.g. ["starter-kit", "use-case:email-agent"]

    def __init__(self, use_api: bool = False):
        self.use_api = use_api
        os.environ["LANGSMITH_PROJECT"] = self.project_name

    @abstractmethod
    def setup_prompts(self) -> None: ...

    @abstractmethod
    def setup_datasets(self) -> None: ...

    @abstractmethod
    def setup_evaluators(self) -> None: ...

    @abstractmethod
    def setup_experiments(self) -> None: ...

    @abstractmethod
    def create_traces(self) -> None: ...

    def setup_annotations(self) -> None:
        pass  # Override if needed

    def run(self, admin: bool = False) -> None:
        setup_project(self.project_name, self.tags)
        print()
        if admin:
            setup_secrets()
        else:
            print("Skipping workspace secret setup (not running with --admin).")
        print()
        self.setup_prompts()
        print()
        self.setup_datasets()
        print()
        self.setup_evaluators()
        time.sleep(3)
        print()
        self.setup_experiments()
        print()
        self.setup_annotations()
        print()
        time.sleep(3)
        self.create_traces()
        print()
