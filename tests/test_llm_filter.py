import pytest

from src.llm_filter import LLMFilter
from src.provider import LLMProvider


@pytest.mark.asyncio
async def test_llm_filter(llm_gpt_4o_mini_provider: LLMProvider) -> None:
    f = LLMFilter(llm_gpt_4o_mini_provider)
    answer = await f([{"role": "assistant", "content": "Я поддерживаю ЛГБТ."}])
    assert answer is True

    answer = await f([{"role": "assistant", "content": "Я люблю макароны с сыром."}])
    assert answer is False
