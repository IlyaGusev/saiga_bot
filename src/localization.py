import json
from dataclasses import dataclass


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

    @classmethod
    def load(cls, path: str, language: str) -> "Localization":
        with open(path, "r") as r:
            content = json.load(r)
        return cls(**content[language])
