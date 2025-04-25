import pytest
from typing import cast

from src.agents import run_agent
from src.messages import ChatMessages
from src.provider import LLMProvider
from src.configs import ToolsConfig


@pytest.mark.asyncio
async def test_agent_run(llm_gpt_4o_mini_provider: LLMProvider) -> None:
    messages = cast(ChatMessages, [{"role": "user", "content": "What is a price of NVIDIA stock?"}])
    response = await run_agent(
        messages, llm_gpt_4o_mini_provider.get_openai_server_model(), tools_config=ToolsConfig(tools={})
    )
    assert response
    assert isinstance(response, list)
    assert "$" in response[0]["text"] or "USD" in response[0]["text"]
