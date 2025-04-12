import pytest

from src.tools.search import WebSearchTool


@pytest.mark.asyncio
async def test_search_tool() -> None:
    tool = WebSearchTool()
    result = tool.forward("What is the capital of France?")
    assert result
    assert "Paris" in result
