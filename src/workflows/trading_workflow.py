from __future__ import annotations

from typing import List, TypedDict

from langgraph.graph import END, START, StateGraph

from ..agents.base_agent import BaseAgent
from ..agents.schemas import AgentMessage


class ConversationState(TypedDict):
    messages: List[AgentMessage]


def build_basic_workflow(agent_a: BaseAgent, agent_b: BaseAgent):
    """Create a simple two-agent conversation workflow."""

    def run_agent_a(state: ConversationState):
        msg = state["messages"][-1]
        reply = agent_a.process_message(msg)
        return {"messages": state["messages"] + [reply]}

    def run_agent_b(state: ConversationState):
        msg = state["messages"][-1]
        reply = agent_b.process_message(msg)
        return {"messages": state["messages"] + [reply]}

    builder = StateGraph(ConversationState)
    builder.add_node("agent_a", run_agent_a)
    builder.add_node("agent_b", run_agent_b)
    builder.add_edge(START, "agent_a")
    builder.add_edge("agent_a", "agent_b")
    builder.add_edge("agent_b", END)

    return builder.compile()
