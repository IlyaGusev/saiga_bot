import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional

from dataclasses_json import DataClassJsonMixin


@dataclass
class SubConfig:
    price: int = 500
    currency: str = "RUB"
    duration: int = 7 * 86400


class SubKey(str, Enum):
    RUB_WEEK = "rub_week"
    RUB_MONTH = "rub_month"
    XTR_WEEK = "xtr_week"
    XTR_MONTH = "xtr_month"


@dataclass
class BotConfig(DataClassJsonMixin):
    token: str
    admin_user_name: str
    admin_user_id: int
    temperature_range: List[float]
    top_p_range: List[float]
    freq_penalty_range: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.1, 0.2, 0.5, 1.0])
    timezone: str = "Europe/Moscow"
    output_chunk_size: int = 3500
    sub_configs: Dict[SubKey, SubConfig] = field(
        default_factory=lambda: {
            SubKey.RUB_WEEK: SubConfig(700, "RUB", 7 * 86400),
            SubKey.RUB_MONTH: SubConfig(2800, "RUB", 31 * 86400),
            SubKey.XTR_WEEK: SubConfig(500, "XTR", 7 * 86400),
            SubKey.XTR_MONTH: SubConfig(1500, "XTR", 31 * 86400),
        }
    )


DEFAULT_HISTORY_MAX_TOKENS = 6144
DEFAULT_MESSAGE_COUNT_LIMIT = {
    "standard": {"limit": 1000, "interval": 86400},
    "subscribed": {"limit": 1000, "interval": 86400},
}
DEFAULT_PARAMS = {
    "temperature": 0.6,
    "top_p": 0.9,
    "max_tokens": 1536,
}


@dataclass
class ProviderConfig(DataClassJsonMixin):
    model_name: str
    base_url: str
    api_key: str
    can_handle_images: bool = False
    can_handle_tools: bool = False
    system_prompt: str = ""
    history_max_tokens: int = DEFAULT_HISTORY_MAX_TOKENS
    params: Dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_PARAMS))
    limits: Dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_MESSAGE_COUNT_LIMIT))
    tokenizer_name: Optional[str] = None
    merge_system: bool = False
    merge_spaces: bool = False

    def __post_init__(self) -> None:
        assert "standard" in self.limits
        assert "subscribed" in self.limits


@dataclass
class ProvidersConfig(DataClassJsonMixin):
    providers: Dict[str, ProviderConfig]


@dataclass
class CharacterConfig(DataClassJsonMixin):
    short_name: str
    public_name: str
    system_prompt: str


@dataclass
class CharactersConfig(DataClassJsonMixin):
    characters: Dict[str, CharacterConfig]
