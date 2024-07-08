import base64
import json
from typing import List, Any, Dict, Union

import requests
from openai import AsyncOpenAI, BadRequestError

from src.tools.base import Tool


@Tool.register("dalle")
class DalleTool(Tool):
    def __init__(self, **kwargs: Any):
        self.client = AsyncOpenAI(**kwargs)

    def get_specification(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "dalle",
                "description": "When there is an explicit command to generate/draw an image,"
                "create a prompt that dalle can use to generate the image.\n"
                "The prompt sent to dalle should be very detailed, and around 50 words long.\n"
                "Only call this tool if the user explicitly asked to generate the image the last message.\n"
                "Never call this tool when you are asked to analyze the image.\n"
                "If you see <image_placeholder> in the last message, do not call this tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Prompt for image generator in English",
                        },
                        "prompt_russian": {
                            "type": "string",
                            "description": "Translation of the prompt to Russian",
                        },
                    },
                    "required": ["prompt", "prompt_russian"],
                },
            },
        }

    async def __call__(self, prompt: str, prompt_russian: str) -> Union[str, List[Dict[str, Any]]]:
        try:
            response = await self.client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
        except BadRequestError as e:
            return json.dumps(e.response.json()["error"]["message"])
        image_url = response.data[0].url
        assert image_url
        encoded_image = self.encode_image(image_url)
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": prompt_russian},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
            },
        ]
        return content

    def encode_image(self, image_url: str) -> str:
        response = requests.get(image_url)
        response.raise_for_status()
        image_data = response.content
        encoded_image = base64.b64encode(image_data).decode("utf-8")
        return encoded_image
