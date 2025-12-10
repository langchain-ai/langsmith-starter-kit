import time
import argparse
from setup.config import setup_secrets, setup_project
from setup.prompts import load_all_prompts
from setup.traces import create_traces
from setup.datasets import load_datasets
from setup.evaluators import load_evaluators
from setup.experiments import load_experiments
from setup.annotations import load_automations_and_queues


def main():
    parser = argparse.ArgumentParser(description="LangSmith Starter Kit")
    parser.add_argument("--api-only", action="store_true", help="Use LangSmith REST API instead of SDK where supported.")
    parser.add_argument("--admin", action="store_true", help="Run with admin privileges (setup workspace secrets).")
    args = parser.parse_args()

    use_api = args.api_only

    setup_project()
    print()
    if args.admin:
        setup_secrets()
    else:
        print("Skipping workspace secret setup (not running with --admin).")
    print()
    load_all_prompts(use_api=use_api)
    print()
    load_datasets(use_api=use_api)
    print()
    load_evaluators(use_api=use_api)

    time.sleep(3)

    print()
    load_experiments(use_api=use_api)
    print()
    load_automations_and_queues()
    print()

    time.sleep(3)

    create_traces()
    print()

if __name__ == "__main__":
    main()