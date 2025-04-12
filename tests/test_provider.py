from typing import cast
import pytest

from src.messages import ChatMessages
from src.provider import LLMProvider


@pytest.mark.asyncio
async def test_provider_base(llm_gpt_4o_mini_provider: LLMProvider) -> None:
    messages = cast(ChatMessages, [{"role": "user", "content": "Respond only with 'Hello, world!'"}])
    response = await llm_gpt_4o_mini_provider(messages)
    assert response
    assert len(response) > 0
    assert "Hello, world!" in response


@pytest.mark.asyncio
async def test_provider_with_system_prompt(llm_gpt_4o_mini_provider: LLMProvider) -> None:
    messages = cast(
        ChatMessages,
        [{"role": "system", "content": "Always respond with 'Hello, world!'"}, {"role": "user", "content": "Hi!"}],
    )
    response = await llm_gpt_4o_mini_provider(messages)
    assert response
    assert len(response) > 0
    assert "Hello, world!" in response


@pytest.mark.asyncio
async def test_provider_count_tokens(llm_gpt_4o_mini_provider: LLMProvider) -> None:
    messages = cast(
        ChatMessages,
        [
            {"role": "user", "content": "Hello, world!"},
            {"role": "assistant", "content": "Hello, world!"},
            {"role": "user", "content": "Hello, world!"},
        ],
    )
    tokens_count = llm_gpt_4o_mini_provider.count_tokens(messages)
    assert tokens_count > 10


@pytest.mark.asyncio
async def test_provider_prepare_history(llm_gpt_4o_mini_provider: LLMProvider) -> None:
    messages = cast(
        ChatMessages,
        [
            {"role": "user", "content": "Hello, world!"},
            {"role": "user", "content": "Hello, world!"},
            {"role": "user", "content": "Hello, world!"},
            {"role": "assistant", "content": "Hello, world!"},
            {"role": "user", "content": "Hello, world!"},
        ],
    )
    history = llm_gpt_4o_mini_provider.prepare_history(messages)
    assert len(history) == 3


@pytest.mark.asyncio
async def test_provider_max_tokens(llm_gpt_4o_mini_provider: LLMProvider) -> None:
    messages = cast(
        ChatMessages,
        [
            {"role": "user", "content": "Hello, world!"},
            {"role": "assistant", "content": "Hello, world!"},
        ],
    )
    messages = messages * 4500
    messages = llm_gpt_4o_mini_provider.prepare_history(messages)
    tokens_count = llm_gpt_4o_mini_provider.count_tokens(messages)
    assert tokens_count <= llm_gpt_4o_mini_provider.config.history_max_tokens
