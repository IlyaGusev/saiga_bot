from duckduckgo_search import AsyncDDGS

from src.tools.base import Tool


@Tool.register("search")
class SearchTool(Tool):
    def __init__(self, proxy: str = None):
        self.proxy = proxy

    async def __call__(self, query: str) -> str:
        client = AsyncDDGS(proxy=None)
        results = await client.atext(query, max_results=5, safesearch="off", backend="html", region="ru-ru")
        snippets = [r["body"] for r in results]
        context = "\n\n".join(snippets)
        return context

    def get_specification(self):
        return {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Use this tool whenever a search query is needed to answer user queries.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Query for search",
                        },
                    },
                    "required": ["query"],
                },
            },
        }
