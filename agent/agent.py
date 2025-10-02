from typing import Literal, TypedDict
from pydantic import BaseModel, Field
from datetime import datetime

from langchain_openai import ChatOpenAI

from agent.tools import get_tools, get_tools_by_name
from setup.prompts import get_triage_instructions, get_action_instructions
from agent.utils import parse_email, format_email_markdown

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command
from dotenv import load_dotenv

load_dotenv(".env")

# Get tools
tools = get_tools()
tools_by_name = get_tools_by_name(tools)

# Initialize the LLM for use with router / structured output
llm = ChatOpenAI(model="gpt-4.1", temperature=0.0)

class RouterSchema(BaseModel):
    """Analyze the unread email and route it according to its content."""

    reasoning: str = Field(
        description="Step-by-step reasoning behind the classification."
    )
    classification: Literal["ignore", "respond", "notify"] = Field(
        description="The classification of an email: 'ignore' for irrelevant emails, "
        "'notify' for important information that doesn't need a response, "
        "'respond' for emails that need a reply",
    )

llm_router = llm.with_structured_output(RouterSchema) 

# Initialize the LLM, enforcing tool use (of any available tools) for agent
llm = ChatOpenAI(model="gpt-4.1", temperature=0.0)
llm_with_tools = llm.bind_tools(tools, tool_choice="any", parallel_tool_calls=False)


# State definitions
class StateInput(TypedDict):
    # This is the input to the state
    email_input: dict

class State(MessagesState):
    # This state class has the messages key build in
    email_input: dict
    classification_decision: Literal["ignore", "respond", "notify"]


# Nodes
def llm_call(state: State):
    """LLM decides whether to call a tool or not"""
    agent_system_prompt = get_action_instructions()
    return {
        "messages": [
            llm_with_tools.invoke([
                {"role": "system", "content": agent_system_prompt.format(today=datetime.now().strftime("%Y-%m-%d"))}
            ] + state["messages"])
        ]
    }

def tool_node(state: State):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append({"role": "tool", "content" : observation, "tool_call_id": tool_call["id"]})
    return {"messages": result}

# Conditional edge function
def should_continue(state: State) -> Literal["Action", "__end__"]:
    """Route to Action, or end if Done tool called"""
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        for tool_call in last_message.tool_calls: 
            if tool_call["name"] == "Done":
                return END
            else:
                return "Action"

# Build workflow
agent_builder = StateGraph(State)

# Add nodes
agent_builder.add_node("agent", llm_call)
agent_builder.add_node("tools", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "agent")
agent_builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        # Name returned by should_continue : Name of next node to visit
        "Action": "tools",
        END: END,
    },
)
agent_builder.add_edge("tools", "agent")

# Compile the agent
agent = agent_builder.compile()


def triage_router(state: State) -> Command[Literal["response_agent", "__end__"]]:
    """
    Analyze email content to decide if we should respond, notify, or ignore.
    """
    author, to, subject, email_thread = parse_email(state["email_input"])
    system_prompt = get_triage_instructions()

    user_prompt = """
Please determine how to handle the below email thread:

From: {author}
To: {to}
Subject: {subject}
{email_thread}""".format(
        author=author, to=to, subject=subject, email_thread=email_thread
    )

    # Create email markdown for Agent Inbox in case of notification  
    email_markdown = format_email_markdown(subject, author, to, email_thread)

    # Run the router LLM
    result = llm_router.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    # Decision
    classification = result.classification

    if classification == "respond":
        goto = "response_agent"
        # Add the email to the messages
        update = {
            "classification_decision": result.classification,
            "messages": [{"role": "user",
                            "content": f"Respond to the email: {email_markdown}"
                        }],
        }
    elif result.classification == "ignore":
        update =  { "classification_decision": result.classification}
        goto = END
    elif result.classification == "notify":
        update = { "classification_decision": result.classification}
        goto = END
    else:
        raise ValueError(f"Invalid classification: {result.classification}")
    return Command(goto=goto, update=update)

# Build workflow
overall_workflow = (
    StateGraph(State, input=StateInput)
    .add_node(triage_router)
    .add_node("response_agent", agent)
    .add_edge(START, "triage_router")
)
email_assistant = overall_workflow.compile()