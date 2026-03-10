"""Set dummy env vars so top-level model imports don't fail during tests."""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LANGSMITH_API_KEY", "test-key")
