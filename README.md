## Setup

1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Create a .env file

```bash
cp .env.example .env
# Open .env and fill in the required values
```

3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Load LangSmith Assets

```bash
python main.py
```


## (Optional) Self-Guided Notebooks

In the root directory, run
```bash
jupyter notebook
```

This will open a web browser page. The self guided notebooks will be located in the ```notebooks``` directory.

Open a notebook and follow along the cells to execute!