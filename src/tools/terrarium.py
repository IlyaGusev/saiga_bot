import json
import aiohttp

from src.tools.base import Tool


@Tool.register("terrarium")
class TerrariumTool(Tool):
    def __init__(self, url: str):
        self.url = url

    async def __call__(self, code: str) -> str:
        data = {"code": code}

        async with aiohttp.ClientSession() as session:
            async with session.post(self.url, json=data) as resp:
                if resp.status != 200:
                    return json.dumps(
                        {
                            "success": False,
                            "error": {"type": "HTTPError", "message": "Error: {result.status_code} - {result.text}"},
                            "std_out": "",
                            "std_err": "",
                            "code_runtime": 0,
                        },
                        ensure_ascii=False,
                    )
                return await resp.text()

    def get_specification(self):
        return {
            "type": "function",
            "function": {
                "name": "terrarium",
                "description": "Calling Python interperter with an isolated environment. "
                "Use only when you need to calculate something or when explicitly asked. "
                "Do not use this tool when code excecution is not actually needed",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute.",
                        },
                    },
                    "required": ["code"],
                },
            },
        }
