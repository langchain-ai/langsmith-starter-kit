import time
from setup.config import setup_secrets
from setup.prompts import load_all_prompts
from setup.traces import create_traces
from setup.datasets import load_datasets
from setup.evaluators import load_evaluators
from setup.experiments import load_experiments
from setup.annotations import load_automations_and_queues


def main():
    print()
    setup_secrets()
    print()
    load_all_prompts()
    print()
    load_datasets()
    print()
    load_evaluators()

    time.sleep(3)

    print()
    load_experiments()
    print()
    load_automations_and_queues()
    print()

    time.sleep(3)

    create_traces()
    print()

if __name__ == "__main__":
    main()