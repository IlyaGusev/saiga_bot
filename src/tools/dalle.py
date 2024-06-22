import base64
from typing import List, Any

import requests
from openai import AsyncOpenAI

from src.tools.base import Tool


@Tool.register("dalle")
class DalleTool(Tool):
    def __init__(self, **kwargs):
        self.client = AsyncOpenAI(**kwargs)

    def get_specification(self):
        return {
            "type": "function",
            "function": {
                "name": "dalle",
                "description": "When there is an explicit command to generate/draw an image,"
                "create a prompt that dalle can use to generate the image.\n"
                "The prompt sent to dalle should be very detailed, and around 50 words long.\n"
                "Only call this tool if the user explicitly asked to draw the image the last message.\n"
                "If you see <image_placeholder>, it means the image was already generated.",
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

    async def __call__(self, prompt: str, prompt_russian: str) -> str:
        response = await self.client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        encoded_image = self.encode_image(image_url)
        content = [
            {"type": "text", "text": prompt_russian},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
            },
        ]
        return content

    def encode_image(self, image_url: str):
        response = requests.get(image_url)
        response.raise_for_status()
        image_data = response.content
        encoded_image = base64.b64encode(image_data).decode("utf-8")
        return encoded_image
