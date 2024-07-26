import json
import pathlib
from dataclasses import dataclass

from jinja2 import Template  # type: ignore


DIR_PATH = pathlib.Path(__file__).parent.resolve()
ROOT_DIR_PATH = DIR_PATH.parent.resolve()
TEMPLATES_DIR_PATH = ROOT_DIR_PATH / "templates"


def get_template(template_name: str) -> Template:
    template_path = TEMPLATES_DIR_PATH / f"{template_name}.jinja"
    with open(template_path) as f:
        return Template(f.read())


@dataclass
class Localization:
    RESET: str
    NO_HISTORY: str
    CHOOSE_MODEL: str
    NEW_MODEL: str
    NEW_SYSTEM_PROMPT: str
    EMPTY_SYSTEM_PROMPT: str
    RESET_SYSTEM_PROMPT: str
    EMPTY_SHORT_NAME: str
    NEW_SHORT_NAME: str
    GET_SHORT_NAME: str
    CHOOSE_CHARACTER: str
    NEW_CHARACTER: str
    HISTORY: str
    INACTIVE_SUB: str
    ACTIVE_SUB: str
    REMAINING_MESSAGES: str
    SET_EMAIL: str
    SUB_TITLE: str
    SUB_SHORT_TITLE: str
    SUB_NOT_CHAT: str
    SUB_SUCCESS: str
    MODEL_NOT_SUPPORTED: str
    LIMIT_EXCEEDED: str
    CLAUDE_HIGH_TEMPERATURE: str
    CONTENT_NOT_SUPPORTED_BY_MODEL: str
    CONTENT_NOT_SUPPORTED: str
    ERROR: str
    HELP: Template
    LIMITS: Template
    SUB_DESCRIPTION: Template
    NEW_TEMPERATURE: str
    NEW_TOP_P: str
    NEW_FREQUENCY_PENALTY: str
    SELECT_TEMPERATURE: str
    SELECT_TOP_P: str
    SELECT_FREQUENCY_PENALTY: str
    CURRENT_PARAMS: str
    TOOLS_NOT_SUPPORTED_BY_MODEL: str
    ENABLED_TOOLS: str
    DISABLED_TOOLS: str
    INCORRECT_EMAIL: str
    FILLED_EMAIL: str
    ADMINS_ONLY: str
    CREATORS_ONLY: str
    PAYMENT_URL: str
    PAYMENT_CANCEL: str
    DALLE_LIMIT: str
    DALLE_PROMPT: str
    DALLE_ERROR: str
    BUY_WITH_STARS: str
    BUY_WITH_RUB: str
    WRONG_COMMAND: str
    PAY_SUPPORT: str
    PRIVACY: Template
    FILE_IS_TOO_BIG: str

    @classmethod
    def load(cls, path: str, language: str) -> "Localization":
        with open(path, "r") as r:
            content = json.load(r)
        lang_content = content[language]
        for k, v in lang_content.items():
            if isinstance(v, dict) and "template_name" in v:
                lang_content[k] = get_template(v["template_name"])
        return cls(**lang_content)
