import pytest

from src.tools.dalle import DalleTool
from src.provider import LLMProvider

from smolagents import AgentImage  # type: ignore


@pytest.mark.asyncio
async def test_dalle_tool(llm_gpt_4o_mini_provider: LLMProvider) -> None:
    tool = DalleTool(llm_gpt_4o_mini_provider.config.api_key)
    result = tool.forward("fox in the forest")
    assert isinstance(result, AgentImage)
