import pytest
from pathlib import Path
from src.provider import LLMProvider
from src.configs import ProvidersConfig

PROVIDERS_CONFIG_PATH = Path("configs/providers.json")


@pytest.fixture
def llm_gpt_4o_mini_provider() -> LLMProvider:
    providers_config = ProvidersConfig.from_json(PROVIDERS_CONFIG_PATH.read_text())
    provider_name = "gpt-4o-mini"
    return LLMProvider(provider_name=provider_name, config=providers_config.providers[provider_name])
