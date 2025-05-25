import pytest

from src.tools.search import WebSearchTool
from src.provider import LLMProvider


@pytest.mark.asyncio
async def test_search_tool(llm_claude_3_5_sonnet_provider: LLMProvider) -> None:
    tool = WebSearchTool(api_key=llm_claude_3_5_sonnet_provider.config.api_key)
    result = tool.forward("What is the capital of France?")
    assert result
    assert "Paris" in result
