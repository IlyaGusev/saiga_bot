import base64
from typing import Any, List

from openai import OpenAI
from smolagents import Tool, AgentImage  # type: ignore

from src.decorators import log_tool_call


@log_tool_call
class EditImageTool(Tool):  # type: ignore
    name = "edit_image"
    description = """
    When there is an explicit command to edit or re-draw an existing image or merge images,
    create a prompt that an image editor can use to edit the image.
    Always provide the prompt only in English!
    The prompt should be as short as possible and it should describe the required change.
    Only call this tool if the user explicitly asked to edit the image in the last message.
    Never call this tool when you are asked to analyze the image.
    Return the image in a final answer like this: `final_answer(image)`.
    Do not return the path to the image or the text, return directly the image itself.
    """
    inputs = {
        "prompt": {
            "type": "string",
            "description": "Short prompt for the image editor in English. Describe only the required change.",
        },
        "image_paths": {
            "type": "array",
            "description": "Paths to the edited images",
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
        self,
        prompt: str,
        image_paths: List[str],
    ) -> AgentImage:
        response = self.client.images.edit(
            model="gpt-image-1",
            prompt=prompt,
            image=[open(p, "rb") for p in image_paths],
        )
        assert response.data, str(response)
        image_base64 = response.data[0].b64_json
        assert image_base64, str(response)
        image_bytes = base64.b64decode(image_base64)
        return AgentImage(image_bytes)
