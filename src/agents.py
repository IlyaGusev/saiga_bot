import asyncio
import base64
from io import BytesIO
from typing import Optional, List, Any, Dict
from smolagents import CodeAgent, AgentImage  # type: ignore
from smolagents.models import OpenAIServerModel  # type: ignore
from PIL import Image

from src.tools import WebSearchTool, VisitWebpageTool, DalleTool
from src.messages import ChatMessages, MessageContent, IMAGE_PLACEHOLDER
from src.utils import get_yaml_prompt, get_jinja_prompt


def encode_image(image: AgentImage) -> str:
    raw = image.to_raw()
    buffer = BytesIO()
    raw.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    encoded_image = base64.b64encode(img_bytes).decode("utf-8")
    return encoded_image


async def run_agent(
    messages: ChatMessages,
    model: OpenAIServerModel,
    max_print_outputs_length: int = 10000,
    verbosity_level: int = 0,
    max_steps: int = 5,
    dalle_api_key: Optional[str] = None,
) -> MessageContent:
    images = []
    for message in messages:
        if not isinstance(message["content"], list):
            continue
        if message["content"][-1]["type"] != "image_url":
            continue
        base64_image = message["content"][-1]["image_url"]["url"].split(",")[-1]
        image_data = base64.b64decode(base64_image)
        image = Image.open(BytesIO(image_data))
        images.append(image)
        message["content"] = message["content"][0]["text"]

    tools = [WebSearchTool(), VisitWebpageTool()]
    if dalle_api_key:
        tools.append(DalleTool(api_key=dalle_api_key))
    agent = CodeAgent(
        tools=tools,
        managed_agents=[],
        model=model,
        add_base_tools=False,
        max_steps=max_steps,
        planning_interval=None,
        verbosity_level=verbosity_level,
        prompt_templates=get_yaml_prompt("agent_prompt"),
        max_print_outputs_length=max_print_outputs_length,
    )
    query_prompt = get_jinja_prompt("agent_query_prompt")
    query = query_prompt.render(conversation=messages)

    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(None, agent.run, query, False, True, images)
    if isinstance(response, AgentImage):
        encoded_image = encode_image(response)
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": IMAGE_PLACEHOLDER},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
            },
        ]
        return content
    return [{"type": "text", "text": str(response)}]
