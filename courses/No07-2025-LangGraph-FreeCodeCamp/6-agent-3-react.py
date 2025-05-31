import os
from typing import List, TypedDict, Union, Annotated, Sequence
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int, b: int) -> int:
    """This is an addition function that adds 2 numbers together."""
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """This is a subtract function that subtracts 2 numbers."""
    return a - b

@tool
def multiply(a: int, b: int) -> int:
    """This is a multiply function that multiplies 2 numbers together."""
    return a * b

tools = [add, subtract, multiply]

model = ChatOpenAI(model="gpt-4o").bind_tools(tools=tools)

def model_call(state: AgentState) -> AgentState:
    """this node will solve the request you input"""
    system_prompt = SystemMessage("You are my AI assistant, please answer my query to the best of your ability.")
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tool_node", tool_node)

graph.set_entry_point("our_agent")
graph.add_conditional_edges("our_agent", should_continue, {
    "continue": "tool_node",
    "end": END,
})
graph.add_edge("tool_node", "our_agent")

agent = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Add 40 + 12 and then multiply the result by 6. Also tell me a joke please")]}
print_stream(agent.stream(inputs, stream_mode="values"))