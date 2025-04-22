import copy
from typing import Dict, Any, cast, Optional

import tiktoken
from openai import AsyncOpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from smolagents.models import OpenAIServerModel  # type: ignore

from src.configs import ProviderConfig
from src.messages import IMAGE_PLACEHOLDER, ChatMessages, format_chat, is_image_content, merge_messages
from src.tokenizers import Tokenizers


class LLMProvider:
    def __init__(
        self,
        provider_name: str,
        config: ProviderConfig,
    ):
        self.config = config
        self.provider_name = provider_name

        self.api = AsyncOpenAI(base_url=self.config.base_url, api_key=self.config.api_key)

    async def __call__(
        self,
        messages: ChatMessages,
        system_prompt: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        system_prompt = self.config.system_prompt if system_prompt is None else system_prompt

        if messages[0]["role"] != "system" and system_prompt.strip():
            messages.insert(0, {"role": "system", "content": system_prompt})

        if self.config.merge_system and messages[0]["role"] == "system":
            system_message = messages[0]["content"]
            assert isinstance(system_message, str)
            messages = messages[1:]
            if isinstance(messages[0]["content"], str):
                messages[0]["content"] = system_message + "\n\n" + messages[0]["content"]
            else:
                messages[0]["content"].insert(0, {"type": "text", "text": system_message})

        casted_messages = [cast(ChatCompletionMessageParam, message) for message in messages]
        chat_completion = await self.api.chat.completions.create(
            *args,
            messages=casted_messages,
            model=self.config.model_name,
            **kwargs,
        )
        assert chat_completion.choices, str(chat_completion)
        content = chat_completion.choices[0].message.content
        reasoning = getattr(chat_completion.choices[0].message, "reasoning", None)
        response_message: str = ""
        if content and not reasoning:
            assert isinstance(content, str), str(chat_completion)
            response_message = content
        elif reasoning and not content:
            response_message = reasoning
        else:
            assert reasoning, str(chat_completion)
            assert content, str(chat_completion)
            response_message = f"Reasoning:\n{reasoning}\n\nAnswer:\n{content}"

        if self.config.merge_spaces:
            response_message = response_message.replace("  ", " ")
        return response_message

    def get_openai_server_model(self) -> OpenAIServerModel:
        model_params: Dict[str, Any] = copy.deepcopy(self.config.params)
        model_params["max_tokens"] = 8192
        return OpenAIServerModel(
            model_id=self.config.model_name,
            api_key=self.config.api_key,
            api_base=self.config.base_url,
            **model_params,
        )

    def can_handle_images(self) -> bool:
        return self.config.can_handle_images

    def can_handle_tools(self) -> bool:
        return self.config.can_handle_tools

    def count_tokens(self, messages: ChatMessages) -> int:
        tokenizer_name = self.config.tokenizer_name
        if not tokenizer_name:
            tokenizer_name = self.config.model_name
        url = str(self.config.base_url)
        tokens_count = 0

        if "api.openai.com" in url:
            encoding = tiktoken.encoding_for_model(tokenizer_name)
            for m in messages:
                if isinstance(m["content"], str):
                    tokens_count += len(encoding.encode(m["content"]))
                elif is_image_content(m["content"]):
                    tokens_count += 2000
                else:
                    tokens_count += len(encoding.encode(m["content"][0]["text"]))
            return tokens_count

        if "anthropic" in url:
            for m in messages:
                if isinstance(m["content"], str):
                    tokens_count += len(m["content"]) // 2
            return tokens_count

        fixed_messages = []
        for m in messages:
            if isinstance(m["content"], str):
                fixed_messages.append(m)
            elif is_image_content(m["content"]):
                fixed_messages.append({"role": m["role"], "content": IMAGE_PLACEHOLDER})
                tokens_count += 2000
            else:
                fixed_messages.append({"role": m["role"], "content": m["content"][0]["text"]})

        tokenizer = Tokenizers.get(tokenizer_name)
        tokens = tokenizer.apply_chat_template(fixed_messages, add_generation_prompt=True)  # type: ignore
        tokens_count += len(tokens)
        return tokens_count

    def prepare_history(self, history: ChatMessages, is_chat: bool = False) -> ChatMessages:
        if is_chat:
            history = format_chat(history)
            assert history

        save_keys = ("content", "role")
        history = [{k: m[k] for k in save_keys if m.get(k) is not None or k == "content"} for m in history]
        history = [m for m in history if not is_image_content(m["content"]) or self.can_handle_images()]
        assert history

        history = merge_messages(history)
        assert history

        max_tokens = self.config.history_max_tokens
        tokens_count = self.count_tokens(history)

        while tokens_count > max_tokens and len(history) >= 3:
            tokens_count -= self.count_tokens(history[:2])
            history = history[2:]

        last_message = history[-1]["content"]
        if tokens_count > max_tokens and isinstance(last_message, str):
            estimated_char_count = max_tokens * 2
            if len(last_message) > estimated_char_count:
                history[-1]["content"] = "<truncated text>\n\n" + last_message[-estimated_char_count:]
        assert history

        history = merge_messages(history)
        assert history

        return history
