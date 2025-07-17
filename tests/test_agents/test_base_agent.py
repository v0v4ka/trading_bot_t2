from src.agents.base_agent import BaseAgent
from src.agents.schemas import AgentMessage


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


def test_base_agent_process_message():
    agent = BaseAgent("test", "system", client=FakeOpenAI("reply"))
    msg = AgentMessage(sender="user", content="hello")
    response = agent.process_message(msg)
    assert response.sender == "test"
    assert response.content == "reply"
