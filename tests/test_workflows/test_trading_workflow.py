from src.agents.base_agent import BaseAgent
from src.agents.schemas import AgentMessage
from src.workflows.trading_workflow import build_basic_workflow


class FakeOpenAI:
    def __init__(self, response: str):
        self.response = response
        self.chat = self.Chat(self)

    class Chat:
        def __init__(self, outer):
            self.completions = outer.Completions(outer)

    class Completions:
        def __init__(self, outer):
            self.outer = outer

        def create(self, *args, **kwargs):
            class Message:
                content = self.outer.response

            class Choice:
                message = Message

            class Response:
                choices = [Choice]

            return Response()


def test_workflow_invocation():
    agent_a = BaseAgent("A", "sys", client=FakeOpenAI("reply_a"))
    agent_b = BaseAgent("B", "sys", client=FakeOpenAI("reply_b"))
    workflow = build_basic_workflow(agent_a, agent_b)
    state = {"messages": [AgentMessage(sender="user", content="start")]}
    result = workflow.invoke(state)
    assert len(result["messages"]) == 3
    assert result["messages"][-1].content == "reply_b"
