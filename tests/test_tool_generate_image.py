import pytest

from src.tools.generate_image import GenerateImageTool
from src.provider import LLMProvider

from smolagents import AgentImage  # type: ignore


@pytest.mark.asyncio
async def test_generate_image_tool(llm_gpt_4o_mini_provider: LLMProvider) -> None:
    tool = GenerateImageTool(api_key=llm_gpt_4o_mini_provider.config.api_key)
    result = tool.forward("fox in the forest")
    assert isinstance(result, AgentImage)
