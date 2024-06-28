from typing import Dict, Any

from trafilatura import fetch_url, extract  # type: ignore

from src.tools.base import Tool


@Tool.register("fetch")
class FetchTool(Tool):
    def __init__(self, max_chars: int = 8000):
        self.max_chars = max_chars

    async def __call__(self, url: str) -> str:
        downloaded = fetch_url(url)
        result = extract(downloaded)
        if result is None:
            return f"Failed to fetch content from url: {url}"
        text: str = result[: self.max_chars]
        return text

    def get_specification(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "fetch",
                "description": "Use this tool to get text content from web pages when a user provides URL.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "Url of the page",
                        },
                    },
                    "required": ["url"],
                },
            },
        }
