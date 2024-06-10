from trafilatura import fetch_url, extract

from src.tools.base import Tool


@Tool.register("fetch")
class FetchTool(Tool):
    def __init__(self, max_chars: int = 8000):
        self.max_chars = max_chars

    async def __call__(self, url: str) -> str:
        downloaded = fetch_url(url)
        result = extract(downloaded)
        return result[:self.max_chars]

    def get_specification(self):
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
