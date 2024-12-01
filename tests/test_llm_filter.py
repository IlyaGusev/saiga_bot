import pytest

from src.llm_filter import LLMFilter


@pytest.mark.asyncio
async def test_llm_filter(llm_gpt_4o_mini_provider):
    f = LLMFilter(llm_gpt_4o_mini_provider)
    answer = await f([{"role": "assistant", "content": "Я поддерживаю ЛГБТ."}])
    assert answer == True

    answer = await f([{"role": "assistant", "content": "Я люблю макароны с сыром."}])
    assert answer == False
