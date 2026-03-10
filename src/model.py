"""Centralized model configuration.

Swap the model string below to use a different provider or model:
    Azure OpenAI:  init_chat_model("azure_openai:gpt-4.1-mini", azure_deployment="gpt-4.1-mini")
    Bedrock:       init_chat_model("bedrock:anthropic.claude-3-5-sonnet-20241022-v2:0")
    Vertex AI:     init_chat_model("vertexai:gemini-2.0-flash")
"""
from langchain.chat_models import init_chat_model

model = init_chat_model("openai:gpt-4.1-mini", temperature=0)

# Eval model — embedded in hub prompts for LLM-judge evaluators.
# Defaults to None, which lets LangSmith use its built-in gpt-5-mini config
# (recommended). Set this only if your workspace needs a custom or internal
# OpenAI-compatible endpoint for evaluations.
# Example: from langchain_openai import ChatOpenAI
#          eval_model = ChatOpenAI(base_url="https://my-proxy/v1", model="gpt-4.1-mini")
eval_model = None
