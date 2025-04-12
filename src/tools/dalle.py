from openai import OpenAI

import requests
from smolagents import Tool, AgentImage  # type: ignore


class DalleTool(Tool):  # type: ignore
    name = "dalle"
    description = """
    When there is an explicit command to generate/draw an image,
    create a prompt that dalle can use to generate the image.
    The prompt sent to dalle should be very detailed, and around 50 words long.
    Only call this tool if the user explicitly asked to generate the image the last message.
    Never call this tool when you are asked to analyze the image.
    Return the image in a final answer like this: `final_answer(image)`.
    Do not return the path to the image, just the image itself.
    """
    inputs = {
        "prompt": {
            "type": "string",
            "description": "Prompt for image generator in English",
        }
    }
    output_type = "image"

    def __init__(self, api_key: str):
        super().__init__()
        self.client = OpenAI(
            base_url="https://api.openai.com/v1",
            api_key=api_key,
        )

    def forward(self, prompt: str) -> AgentImage:
        response = self.client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        assert image_url and isinstance(image_url, str)
        image_response = requests.get(image_url)
        assert image_response
        image_response.raise_for_status()
        image_data = image_response.content
        return AgentImage(image_data)
