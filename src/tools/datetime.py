import os
import json
from typing import Dict
from datetime import datetime

import requests

from src.tools.base import Tool


@Tool.register("datetime")
class DateTimeTool(Tool):
    def __init__(self, default_timezone: str = None):
        self.default_timezone = default_timezone
        if default_timezone is None:
            self.default_timezone = "Europe/Moscow"

    def get_specification(self):
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

    async def __call__(self, timezone: str) -> Dict:
        url = f"https://worldtimeapi.org/api/timezone/{timezone}"

        try:
            wtr = requests.get(url).json().get("datetime")
            wtr_obj = datetime.strptime(wtr, "%Y-%m-%dT%H:%M:%S.%f%z")
            return wtr_obj.strftime("Date: %Y-%m-%d, time: %H:%M:%S")
        except:
            return json.dumps({"result": "No result was found"})
