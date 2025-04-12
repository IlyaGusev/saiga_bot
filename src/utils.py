from typing import Any, Dict
from pathlib import Path

import yaml
from jinja2 import Template

DIR_PATH = Path(__file__).parent
ROOT_PATH = DIR_PATH.parent
PROMPTS_DIR_PATH = ROOT_PATH / "prompts"


def get_yaml_prompt(template_name: str) -> Dict[str, Any]:
    template_path = PROMPTS_DIR_PATH / f"{template_name}.yaml"
    with open(template_path) as f:
        template = f.read()
    templates: Dict[str, Any] = yaml.safe_load(template)
    return templates


def get_jinja_prompt(template_name: str) -> Template:
    template_path = PROMPTS_DIR_PATH / f"{template_name}.jinja2"
    with open(template_path) as f:
        template = Template(f.read())
    return template
