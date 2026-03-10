# LangSmith Starter Kit

A multi-use-case starter kit that provisions LangSmith projects with datasets, evaluators, experiments, annotation queues, and traces in one command.

**Built-in use cases:** `email-agent` · `chatbot`

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Copy and fill in your credentials
cp .env.example .env

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running a Use Case

```bash
# Run the default use case (email-agent)
python main.py

# Run a specific use case
python main.py --use-case chatbot

# Control how many traces to generate
python main.py --use-case chatbot --num-traces 5

# Use the LangSmith REST API instead of the SDK (functionally identical)
python main.py --api-only

# Upload workspace secrets to LangSmith (requires admin access)
python main.py --admin
```

Each run is **idempotent** — existing datasets, evaluators, and queues are skipped automatically. To regenerate traces, re-run with `--num-traces N`.

---

## LangGraph Studio

Both graphs are registered in `langgraph.json` and can be explored interactively:

```bash
langgraph dev
```

---

## Contributing a New Use Case

Adding a new use case takes five steps. All the LangSmith plumbing is handled by helpers in `utils/` — you only write your application logic.

### Step 1 — Create the directory structure

```
src/
└── my_use_case/
    ├── __init__.py
    ├── agent/
    │   ├── __init__.py
    │   └── agent.py          # Your LangGraph graph
    ├── data/                 # CSV / JSONL datasets (optional)
    ├── setup/
    │   ├── __init__.py
    │   ├── datasets.py
    │   ├── evaluators.py
    │   ├── experiments.py
    │   └── traces.py
    └── use_case.py
```

### Step 2 — Write the setup files

Each file follows the same pattern: write your logic, call a helper from `utils/`.

#### `setup/datasets.py`

```python
from utils.datasets import create_langsmith_dataset

def load_datasets(use_api: bool = False) -> None:
    create_langsmith_dataset(
        "My Dataset",
        inputs=[{"question": "What is X?"}, ...],
        outputs=[{"answer": "X is ..."}, ...],
        use_api=use_api,
    )
```

`create_langsmith_dataset` is idempotent — it skips creation if the dataset already exists.

#### `setup/evaluators.py`

**Code evaluator** — write a normal Python function named `perform_eval`:

```python
from utils.evaluators import create_evaluator

def load_evaluators(use_api: bool = False) -> None:
    def perform_eval(run, example):
        # run["outputs"] contains the agent's output
        # example["outputs"] contains the reference output
        correct = run["outputs"]["label"] == example["outputs"]["label"]
        return {"correctness": correct}

    create_evaluator("correctness", "My Dataset", func=perform_eval)
```

> **Important:** The function must be named `perform_eval`. All imports used inside it must be inside the function body (it is serialized and executed remotely by LangSmith).

**LLM judge evaluator** — pass a hub ref or inline prompt:

```python
# Hub ref (must be pushed to LangSmith first)
create_evaluator("quality", "My Dataset",
    prompt_or_ref="my-eval-prompt:latest", score_type="boolean")

# Inline prompt (no hub push needed)
create_evaluator("quality", "My Dataset",
    prompt_or_ref=[
        ["system", "You are an expert evaluator. Score the answer 0 or 1."],
        ["human", "Question: {input}\n\nAnswer: {output}\n\nReference: {reference}"],
    ],
    score_type="boolean",
)
```

#### `setup/experiments.py`

```python
from utils.config import client
from src.my_use_case.agent.agent import my_graph

def _run_agent(inputs: dict) -> dict:
    result = my_graph.invoke(inputs)
    return {"output": result["output"]}

def load_experiments(use_api: bool = False) -> None:
    client.evaluate(
        _run_agent,
        data="My Dataset",
        evaluators=[],           # inline evaluators (optional)
        experiment_prefix="my-use-case",
        max_concurrency=4,
    )
```

#### `setup/traces.py`

```python
from typing import Optional
from src.my_use_case.agent.agent import my_graph

def create_traces(num_traces: Optional[int] = None) -> None:
    inputs = [...]   # load from CSV or define inline
    sample = inputs[:num_traces] if num_traces else inputs
    for inp in sample:
        my_graph.invoke(inp, config={"tags": ["my-use-case", "trace"]})
```

### Step 3 — Create `use_case.py`

```python
from typing import Optional
from src.base import UseCase
from src.my_use_case.setup.datasets import load_datasets
from src.my_use_case.setup.evaluators import load_evaluators
from src.my_use_case.setup.experiments import load_experiments
from src.my_use_case.setup.traces import create_traces as _create_traces

class MyUseCase(UseCase):
    name = "my-use-case"
    project_name = "starter-kit-my-use-case"
    tags = ["starter-kit", "use-case:my-use-case"]

    def setup_prompts(self): pass         # omit if no hub prompts
    def setup_datasets(self): load_datasets(self.use_api)
    def setup_evaluators(self): load_evaluators(self.use_api)
    def setup_experiments(self): load_experiments(self.use_api)
    def create_traces(self, num_traces=None): _create_traces(num_traces=num_traces)
```

`UseCase` also has an optional `setup_annotations()` hook for creating annotation queues and automations — see `src/email_agent/setup/annotations.py` for an example using `utils/annotations.py`.

### Step 4 — Register in `main.py`

```python
from src.my_use_case.use_case import MyUseCase

USE_CASES = {
    "email-agent": EmailAgentUseCase,
    "chatbot": ChatbotUseCase,
    "my-use-case": MyUseCase,          # ← add this line
}
```

### Step 5 — Run it

```bash
python main.py --use-case my-use-case
```

---

## Utils Reference

| Helper | What it does |
|--------|-------------|
| `utils.datasets.create_langsmith_dataset(name, inputs, outputs, use_api)` | Create a dataset if it doesn't exist |
| `utils.evaluators.create_evaluator(name, target, func=...)` | Attach a code evaluator to a dataset or project |
| `utils.evaluators.create_evaluator(name, target, prompt_or_ref=..., score_type=...)` | Attach an LLM judge evaluator |
| `utils.prompts.load_prompt(name, chain, use_api)` | Push a LangChain chain as a prompt to LangSmith Hub |
| `utils.prompts.delete_existing_prompt(name, use_api)` | Delete a prompt (used before re-pushing with new commits) |
| `utils.prompts.build_schema(PydanticModel, field_name)` | Build a JSON schema for `StructuredPrompt` evaluators |
| `utils.annotations.create_queue(name, ...)` | Create an annotation queue |
| `utils.annotations.create_automation(name, project_id, queue_id, filter)` | Route traces matching a filter into a queue |
| `utils.config.get_project_id(name)` | Look up a LangSmith project ID by name |

All helpers are idempotent — they check for existing resources before creating.

---

## Project Structure

```
starter-kit/
├── src/
│   ├── base.py                    # UseCase ABC
│   ├── email_agent/
│   │   ├── agent/                 # LangGraph email assistant
│   │   ├── data/                  # emails.csv, next_action.jsonl
│   │   ├── setup/                 # datasets, evaluators, experiments, traces, annotations
│   │   └── use_case.py
│   └── chatbot/
│       ├── agent/                 # LangGraph ReAct chatbot with KB retrieval
│       ├── data/                  # eval CSVs, question bank, synthetic KB
│       ├── setup/                 # datasets, evaluators, experiments, traces
│       └── use_case.py
├── utils/
│   ├── config.py                  # auth, client, setup_project, get_project_id
│   ├── datasets.py                # create_langsmith_dataset
│   ├── evaluators.py              # create_evaluator
│   ├── experiments.py             # API session/run/feedback helpers
│   ├── prompts.py                 # load_prompt, build_schema, prompt_exists
│   └── annotations.py            # create_queue, create_automation
├── main.py                        # CLI entry point
├── langgraph.json                 # Graph registry for LangGraph Studio
└── requirements.txt
```
