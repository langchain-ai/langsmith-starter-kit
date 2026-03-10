"""Centralized model configuration.

Swap the model string below to use a different provider or model:
    Azure OpenAI:  init_chat_model("azure_openai:gpt-4.1-mini", azure_deployment="gpt-4.1-mini")
    Bedrock:       init_chat_model("bedrock:anthropic.claude-3-5-sonnet-20241022-v2:0")
    Vertex AI:     init_chat_model("vertexai:gemini-2.0-flash")
"""
from langchain.chat_models import init_chat_model

model = init_chat_model("openai:gpt-4.1-mini", temperature=0)

# Eval model — embedded in hub prompts for LLM-judge evaluators.
# Defaults to gpt-5-mini (uses Responses API, handles tool_call content blocks).
# Override with a ChatOpenAI-compatible instance to use a custom or internal endpoint.
# Example: eval_model = ChatOpenAI(base_url="https://my-proxy/v1", model="gpt-4.1-mini")
from langchain_openai import ChatOpenAI as _ChatOpenAI
eval_model = _ChatOpenAI(model="gpt-5-mini", temperature=0)
