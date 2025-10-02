from setup.prompts import load_all_prompts
from setup.traces import create_traces
from setup.datasets import load_datasets


def main():
    load_all_prompts()
    load_datasets()
    create_traces()

if __name__ == "__main__":
    main()