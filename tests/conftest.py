import json
import pytest

from src.provider import LLMProvider

PROVIDERS_CONFIG_PATH = "configs/providers.json"


@pytest.fixture
def llm_gpt_4o_mini_provider():
    with open(PROVIDERS_CONFIG_PATH) as r:
        providers_config = json.load(r)
    provider_name = "gpt-4o-mini"
    return LLMProvider(provider_name=provider_name, **providers_config[provider_name])
