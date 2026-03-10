import os
import time
from abc import ABC, abstractmethod
from typing import Optional

from utils.config import setup_project, setup_secrets, tag_all_resources
from utils.teardown import teardown_use_case


class UseCase(ABC):
    name: str           # slug, e.g. "email-agent"
    project_name: str   # LangSmith project, e.g. "starter-email-agent"
    tags: list          # Application tags, e.g. ["starter-kit", "starter:email-agent"]
    dataset_names: list = []   # Dataset names to delete on teardown
    prompt_names: list = []    # Hub prompt names to delete on teardown
    queue_names: list = []     # Annotation queue names to delete on teardown

    def __init__(self):
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
    def create_traces(self, num_traces: Optional[int] = None) -> None: ...

    def setup_annotations(self) -> None:
        pass  # Override if needed

    def teardown(self) -> None:
        teardown_use_case(
            project_name=self.project_name,
            dataset_names=self.dataset_names,
            prompt_names=self.prompt_names,
            queue_names=self.queue_names,
            tags=self.tags,
        )

    def run(self, admin: bool = False, num_traces: Optional[int] = None) -> None:
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
        tag_all_resources(self.dataset_names, self.queue_names, self.prompt_names, self.tags)
        time.sleep(3)
        self.create_traces(num_traces=num_traces)
        print()
