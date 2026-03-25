from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from src.model import model
from src.finance_qa.agent.tools import search_kb_tool, get_topic_details, list_topics

load_dotenv(".env")

system_prompt = """You are a Knowledge Base Specialist that retrieves and synthesizes information from a customer support knowledge base.

## Your Mission

Search the knowledge base to find relevant information for customer questions. Your KB contains banking and credit card support procedures covering topics like:
- Payment processing and methods
- Disputes and chargebacks
- Rewards programs and redemption
- Card activation and replacement
- Fraud protection and monitoring
- Balance transfers
- Credit limit changes
- Account statements
- Interest rates and fees
- Account closures
- Lost or stolen cards

## Available Tools

Use `search_kb_tool` to search the knowledge base, `get_topic_details` to get complete information about a specific topic, and `list_topics` to browse available topics.

## Research Strategy

1. **Search the KB** using the main concepts from the question
2. **Review results** — if relevant, extract key information; if partial, search related terms
3. **Get detailed information** if you need complete procedures for a topic
4. **Synthesize findings** into a clear, helpful answer

**If the KB doesn't contain relevant information**, explain what you searched for and what you found (or didn't find). Don't speculate beyond what's in the KB.

## Response Format

Provide a clear, helpful answer to the customer's question. At the end of your response, include a "Relevant docs:" section listing the KB sources you used:

Relevant docs:
- [Topic name 1]
- [Topic name 2]

If the KB doesn't have relevant information, still include "Relevant docs:" with an empty list and explain what you searched for.
"""

REFUSAL_MESSAGE = (
    "I'm sorry, but I'm unable to process this request. "
    "Please rephrase your question in a respectful manner and I'll be happy to help."
)


class ToxicityCheck(BaseModel):
    """Assess whether a user message contains toxic content."""

    reasoning: str = Field(description="Brief reasoning about the content.")
    is_toxic: bool = Field(
        description="True if the message contains toxic, abusive, threatening, or hateful content."
    )


toxicity_classifier = model.with_structured_output(ToxicityCheck)

TOXICITY_PROMPT = (
    "You are a content moderation classifier for a banking customer support system. "
    "Determine whether the following user message contains toxic, abusive, threatening, "
    "or hateful content. Legitimate complaints, frustration about banking issues, or "
    "strong but non-abusive language should NOT be flagged as toxic.\n\n"
    "User message: {message}"
)


agent = create_react_agent(
    model,
    tools=[search_kb_tool, get_topic_details, list_topics],
    prompt=system_prompt,
)


def toxicity_guardrail(state: MessagesState) -> Command[Literal["agent", "__end__"]]:
    """Check the latest user message for toxic content."""
    last_human = None
    for msg in reversed(state["messages"]):
        if msg.type == "human":
            last_human = msg.content
            break

    if last_human is None:
        return Command(goto="agent")

    result = toxicity_classifier.invoke(
        TOXICITY_PROMPT.format(message=last_human)
    )

    if result.is_toxic:
        return Command(
            goto=END,
            update={"messages": [AIMessage(content=REFUSAL_MESSAGE)]},
        )

    return Command(goto="agent")


memory = MemorySaver()

chatbot = (
    StateGraph(MessagesState)
    .add_node("toxicity_guardrail", toxicity_guardrail)
    .add_node("agent", agent)
    .add_edge(START, "toxicity_guardrail")
    .add_edge("agent", END)
    .compile(checkpointer=memory)
)
