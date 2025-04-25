from smolagents.default_tools import VisitWebpageTool  # type: ignore

from src.tools.search import WebSearchTool
from src.tools.generate_image import GenerateImageTool
from src.tools.edit_image import EditImageTool

TOOLS = dict()

for tool_cls in (WebSearchTool, GenerateImageTool, VisitWebpageTool, EditImageTool):
    TOOLS[tool_cls.name] = tool_cls


__all__ = ["WebSearchTool", "VisitWebpageTool", "GenerateImageTool", "EditImageTool", "TOOLS"]
