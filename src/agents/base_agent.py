from __future__ import annotations

from typing import Optional

from openai import OpenAI

from .schemas import AgentMessage


class BaseAgent:
    """Base class for GPT-powered agents."""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        model: str = "gpt-4o",
        client: Optional[OpenAI] = None,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
        # Client can be injected for testing
        self.client = client or OpenAI()

    def process_message(self, message: AgentMessage) -> AgentMessage:
        """Generate a reply to the given message using the LLM."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": message.content},
            ],
        )
        reply = response.choices[0].message.content
        return AgentMessage(sender=self.name, content=reply)
