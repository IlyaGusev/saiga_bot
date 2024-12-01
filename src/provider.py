import copy
from typing import Dict, Any, Optional

from openai import AsyncOpenAI

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


class LLMProvider:
    def __init__(
        self,
        provider_name: str,
        model_name: str,
        base_url: str,
        api_key: str,
        can_handle_images: bool = False,
        can_handle_tools: bool = False,
        system_prompt: str = "",
        history_max_tokens: int = DEFAULT_HISTORY_MAX_TOKENS,
        params: Dict[str, Any] = DEFAULT_PARAMS,
        message_count_limit: Dict[str, Any] = DEFAULT_MESSAGE_COUNT_LIMIT,
        tokenizer_name: Optional[str] = None,
        merge_system: bool = False,
        merge_spaces: bool = False,
    ):
        self.provider_name = provider_name
        self.model_name = model_name
        self.can_handle_images = can_handle_images
        self.can_handle_tools = can_handle_tools
        self.system_prompt = system_prompt
        self.history_max_tokens = history_max_tokens
        self.params = copy.deepcopy(params)
        self.limits = message_count_limit
        self.tokenizer_name = tokenizer_name
        self.merge_system = merge_system
        self.merge_spaces = merge_spaces
        assert "standard" in self.limits
        assert "subscribed" in self.limits
        self.api = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def __call__(self, *args: Any, **kwargs: Any) -> str:
        chat_completion = await self.api.chat.completions.create(
            *args,
            model=self.model_name,
            **kwargs,
        )
        response_message: str = chat_completion.choices[0].message.content
        return response_message
