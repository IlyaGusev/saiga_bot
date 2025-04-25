import base64
from typing import Any

from openai import OpenAI
from smolagents import Tool, AgentImage  # type: ignore

from src.decorators import log_tool_call


QUALITIES = ("low", "medium", "high")
SIZES = ("1024x1024", "1536x1024", "1024x1536")
BACKGROUNDS = ("auto", "transparent", "opaque")


@log_tool_call
class GenerateImageTool(Tool):  # type: ignore
    name = "generate_image"
    description = """
    When there is an explicit command to generate/draw an image,
    create a prompt that an image generator can use to generate the image.
    The prompt sent to the generator should be very detailed, and around 30 words long.
    Only call this tool if the user explicitly asked to generate the image in the last message.
    Never call this tool when you are asked to analyze the image.
    Return the image in a final answer like this: `final_answer(image)`.
    Do not return the path to the image, just the image itself.
    """
    inputs = {
        "prompt": {
            "type": "string",
            "description": "Prompt for the image generator in English",
        },
        "quality": {
            "type": "string",
            "description": "Image quality. 'medium' by default, other options: 'low'",
            "nullable": True,
        },
        "size": {
            "type": "string",
            "description": "Image dimensions. '1024x1024' by default, other options: '1536x1024', '1024x1536'",
            "nullable": True,
        },
        "background": {
            "type": "string",
            "description": "Transparent or opaque. 'auto' by default, other options: 'transparent', 'opaque'",
            "nullable": True,
        },
    }
    output_type = "image"

    def __init__(self, *args: Any, api_key: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.client = OpenAI(
            base_url="https://api.openai.com/v1",
            api_key=api_key,
        )

    def forward(
        self, prompt: str, quality: str = "medium", size: str = "1024x1024", background: str = "auto"
    ) -> AgentImage:
        assert quality in QUALITIES, f"Wrong quality: {quality}! Options: {QUALITIES}"
        assert size in SIZES, f"Wrong size: {size}! Options: {SIZES}"
        assert background in BACKGROUNDS, f"Wrong background option: {background}. Options: {BACKGROUNDS}"

        response = self.client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            quality=quality,  # type: ignore
            size=size,  # type: ignore
            background=background,  # type: ignore
        )
        assert response.data, str(response)
        image_base64 = response.data[0].b64_json
        assert image_base64, str(response)
        image_bytes = base64.b64decode(image_base64)
        return AgentImage(image_bytes)
