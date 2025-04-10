import json
from typing import Dict, Optional, Any
from datetime import datetime

import aiohttp

from src.tools.base import Tool


@Tool.register("datetime")
class DateTimeTool(Tool):
    def __init__(self, default_timezone: Optional[str] = None):
        self.default_timezone = default_timezone
        if default_timezone is None:
            self.default_timezone = "Europe/Moscow"

    def get_specification(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "datetime",
                "description": "Get the current date and time from a given timezone",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "The timezone identifier (e.g: `Europe/Rome`). Infer this from the location."
                            f"Use {self.default_timezone} if not specified.",
                        }
                    },
                    "required": ["timezone"],
                },
            },
        }

    async def __call__(self, timezone: str) -> str:
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("Date: %Y-%m-%d, Time: %H:%M:%S")
        return formatted_datetime
