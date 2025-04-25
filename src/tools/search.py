from typing import Any

from duckduckgo_search import DDGS
from smolagents import Tool  # type: ignore

from src.decorators import log_tool_call


@log_tool_call
class WebSearchTool(Tool):  # type: ignore
    name = "web_search"
    description = """
    Performs a duckduckgo web search based on your query (think a Google search),
    then returns the top search results.
    """
    inputs = {"query": {"type": "string", "description": "The search query to perform."}}
    output_type = "string"

    def __init__(self, *args: Any, max_results: int = 10, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.max_results = max_results
        self.ddgs = DDGS(**kwargs)

    def forward(self, query: str) -> str:
        results = self.ddgs.text(query, max_results=5, safesearch="off", backend="html", region="ru-ru")
        if len(results) == 0:
            raise Exception("No results found! Try a less restrictive/shorter query.")
        postprocessed_results = [f"[{result['title']}]({result['href']})\n{result['body']}" for result in results]
        return "## Search Results\n\n" + "\n\n".join(postprocessed_results)
