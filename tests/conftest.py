import pytest
from pathlib import Path
from src.provider import LLMProvider
from src.configs import ProvidersConfig, ToolsConfig

PROVIDERS_CONFIG_PATH = Path("configs/providers.json")
TOOLS_CONFIG_PATH = Path("configs/tools.json")


@pytest.fixture
def llm_gpt_4o_mini_provider() -> LLMProvider:
    providers_config = ProvidersConfig.from_json(PROVIDERS_CONFIG_PATH.read_text())
    provider_name = "gpt-4o-mini"
    return LLMProvider(provider_name=provider_name, config=providers_config.providers[provider_name])


@pytest.fixture
def llm_claude_3_5_sonnet_provider() -> LLMProvider:
    providers_config = ProvidersConfig.from_json(PROVIDERS_CONFIG_PATH.read_text())
    provider_name = "claude-3-5-sonnet"
    return LLMProvider(provider_name=provider_name, config=providers_config.providers[provider_name])


@pytest.fixture
def tools_config() -> ToolsConfig:
    tools_config = ToolsConfig.from_json(TOOLS_CONFIG_PATH.read_text())
    return tools_config
