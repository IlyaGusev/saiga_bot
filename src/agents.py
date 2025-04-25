import asyncio
import tempfile
import base64
import os
from io import BytesIO
from typing import Optional, List, Any, Dict
from smolagents import CodeAgent, AgentImage  # type: ignore
from smolagents.models import OpenAIServerModel  # type: ignore
from PIL import Image

from src.tools import VisitWebpageTool, TOOLS
from src.messages import ChatMessages, MessageContent, IMAGE_PLACEHOLDER
from src.utils import get_yaml_prompt, get_jinja_prompt
from src.database import Database
from src.configs import ToolsConfig


def encode_image(image: AgentImage) -> str:
    raw = image.to_raw()
    buffer_io = BytesIO()
    raw.save(buffer_io, format="PNG")
    img_bytes = buffer_io.getvalue()
    encoded_image = base64.b64encode(img_bytes).decode("utf-8")
    return encoded_image


async def run_agent(
    messages: ChatMessages,
    model: OpenAIServerModel,
    tools_config: ToolsConfig,
    max_print_outputs_length: int = 10000,
    verbosity_level: int = 0,
    max_steps: int = 10,
    custom_system_prompt: Optional[str] = None,
    db: Optional[Database] = None,
    user_id: Optional[int] = None,
) -> MessageContent:
    images = []
    image_paths = []
    for message in messages:
        if not isinstance(message["content"], list):
            continue
        if message["content"][-1]["type"] != "image_url":
            continue
        base64_image = message["content"][-1]["image_url"]["url"].split(",")[-1]
        if "text" in message["content"][0]:
            message["content"] = message["content"][0]["text"]

        image_data = base64.b64decode(base64_image)
        image = Image.open(BytesIO(image_data))
        images.append(image)

        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image.save(temp_file.name)
        image_paths.append(temp_file.name)

    tools = [VisitWebpageTool()]
    for tool_name, tool_config in tools_config.tools.items():
        tool_cls = TOOLS[tool_name]
        tool_kwargs = tool_config.kwargs
        tools.append(
            tool_cls(
                user_id=user_id,
                db=db,
                limits=tool_config.limits,
                **tool_kwargs,
            )
        )
    prompts = get_yaml_prompt("agent_prompt")
    if custom_system_prompt:
        prompts["system_prompt"] = custom_system_prompt + "\n\n" + prompts["system_prompt"]
    agent = CodeAgent(
        tools=tools,
        managed_agents=[],
        model=model,
        add_base_tools=False,
        max_steps=max_steps,
        planning_interval=None,
        verbosity_level=verbosity_level,
        prompt_templates=prompts,
        max_print_outputs_length=max_print_outputs_length,
    )
    query_prompt = get_jinja_prompt("agent_query_prompt")
    query = query_prompt.render(conversation=messages, image_paths=image_paths)

    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(None, agent.run, query, False, True, images)
    for path in image_paths:
        os.unlink(path)
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
