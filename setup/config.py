import os
import requests
from typing import Dict
from dotenv import load_dotenv
from langsmith import Client

load_dotenv(".env")

client = Client()

LANGSMITH_API_URL = os.getenv("LANGSMITH_API_URL", "https://api.smith.langchain.com")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def auth_headers() -> Dict[str, str]:
    if not LANGSMITH_API_KEY:
        raise RuntimeError("LANGSMITH_API_KEY is required in environment.")
    return {
        "x-api-key": LANGSMITH_API_KEY,
        "Content-Type": "application/json",
    }

def setup_secrets() -> None:
    """
    Upsert workspace secrets (OPENAI_API_KEY) to LangSmith.
    """
    print("Setting up secrets...")
    if not OPENAI_API_KEY:
        print("    - OPENAI_API_KEY not set in environment; skipping workspace secret upsert.")
        return

    url = f"{LANGSMITH_API_URL}/workspaces/current/secrets"
    payload = [{"key": "OPENAI_API_KEY", "value": OPENAI_API_KEY}]

    resp = requests.post(url, headers=auth_headers(), json=payload, timeout=30)
    if resp.status_code >= 300:
        raise RuntimeError(f"Failed to upsert workspace secret OPENAI_API_KEY: {resp.status_code} {resp.text}")
    print(f"    - OPENAI_API_KEY upserted to LangSmith workspace.")