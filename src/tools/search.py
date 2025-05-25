from typing import Any, List, cast

from smolagents import Tool  # type: ignore
from anthropic import Anthropic
from anthropic.types.text_citation import TextCitation
from anthropic.types.citations_web_search_result_location import CitationsWebSearchResultLocation

from src.decorators import log_tool_call


CITATION_FORMAT = """
Source: [{title}]({url})
Snippet: {cited_text}
"""


@log_tool_call
class WebSearchTool(Tool):  # type: ignore
    name = "web_search"
    description = """
    Performs a web search based on your query (think a Google search),
    then returns the top search results.
    """
    inputs = {"query": {"type": "string", "description": "The search query to perform."}}
    output_type = "string"

    def __init__(self, *args: Any, api_key: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.client = Anthropic(api_key=api_key)

    def forward(self, query: str) -> str:
        query += "\nAlways perform a web search for the query, even if the answer is already known."
        response = self.client.messages.create(
            model="claude-3-7-sonnet-latest",
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 1}],
            max_tokens=2048,
            messages=[{"role": "user", "content": query}],
        )
        citations: List[TextCitation] = []
        text = ""
        for block in response.content:
            if hasattr(block, "text") and block.text is not None:
                text += block.text
            if hasattr(block, "citations") and block.citations is not None:
                citations.extend(block.citations)
        unique_citations = {c.cited_text.strip(): c for c in citations}.values()
        citation_texts = []
        for c in unique_citations:
            if not hasattr(c, "cited_text"):
                continue
            citation = cast(CitationsWebSearchResultLocation, c)
            citation_text = CITATION_FORMAT.format(
                title=citation.title, url=citation.url, cited_text=citation.cited_text.strip()
            )
            citation_texts.append(citation_text)
        return f"# Search results\n{'\n'.join(citation_texts)}\n\n# Final response:\n{text}"
