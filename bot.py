import asyncio
import os
import json
import secrets
from datetime import datetime, timezone

import fire
from aiogram import Bot, Dispatcher
from aiogram import F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import Message, InlineKeyboardButton, CallbackQuery
from aiogram.utils.keyboard import InlineKeyboardBuilder
from openai import AsyncOpenAI
from tinydb import TinyDB, where, Query
from tinydb import operations as ops
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."


class Tokenizer:
    tokenizers = dict()

    @classmethod
    def get(cls, model_name: str):
        if model_name not in cls.tokenizers:
            cls.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
        return cls.tokenizers[model_name]


class LlmBot:
    def __init__(
        self,
        bot_token: str,
        api_key: str,
        base_url: str,
        db_path: str,
        model_name: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        history_max_tokens: int
    ):
        # Клиент
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.history_max_tokens = history_max_tokens

        # База
        self.db = TinyDB(db_path, ensure_ascii=False)
        self.messages_table = self.db.table("messages")
        self.conversations_table = self.db.table("current_conversations")
        self.system_prompts_table = self.db.table("system_prompts")
        self.likes_table = self.db.table("likes")

        # Бот
        self.bot = Bot(token=bot_token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
        self.dp = Dispatcher()
        self.dp.message.register(self.start, Command("start"))
        self.dp.message.register(self.reset, Command("reset"))
        self.dp.message.register(self.history, Command("history"))
        self.dp.message.register(self.set_system, Command("set_system"))
        self.dp.message.register(self.get_system, Command("get_system"))
        self.dp.message.register(self.reset_system, Command("reset_system"))
        self.dp.message.register(self.generate)
        self.dp.callback_query.register(self.save_like, F.data == "like")
        self.dp.callback_query.register(self.save_like, F.data == "dislike")

    async def start_polling(self):
        await self.dp.start_polling(self.bot)

    def count_tokens(self, messages):
        tokenizer = Tokenizer.get(self.model_name)
        tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        return len(tokens)

    def fetch_conversation(self, conv_id):
        messages = self.messages_table.search(where("conv_id") == conv_id)
        if not messages:
            return []
        messages.sort(key=lambda x: x["timestamp"])
        return [{"role": m["role"], "content": m["content"]} for m in messages]

    @staticmethod
    def merge_messages(messages):
        new_messages = []
        prev_role = None
        for m in messages:
            if m["role"] == prev_role:
                new_messages[-1]["content"] += "\n" + m["content"]
                continue
            prev_role = m["role"]
            new_messages.append(m)
        return new_messages

    @staticmethod
    def get_current_ts():
        return int(datetime.now().replace(tzinfo=timezone.utc).timestamp())

    def create_conv_id(self, user_id):
        conv_id = secrets.token_hex(nbytes=16)
        self.conversations_table.insert({
            "user_id": user_id,
            "conv_id": conv_id,
            "timestamp": self.get_current_ts()
        })
        return conv_id

    def get_current_conv_id(self, user_id):
        conv_ids = self.conversations_table.search(where("user_id") == user_id)
        if not conv_ids:
            return self.create_conv_id(user_id)
        conv_ids.sort(key=lambda x: x["timestamp"], reverse=True)
        return conv_ids[0]["conv_id"]

    async def query_api(self, history, last_message: str, system_prompt: str):
        messages = history + [{"role": "user", "content": last_message}]
        messages = self.merge_messages(messages)
        tokens_count = self.count_tokens(messages)
        while tokens_count > self.history_max_tokens:
            messages = messages[2:]
            tokens_count = self.count_tokens(messages)

        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": system_prompt})

        print(messages)
        chat_completion = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens
        )
        answer = chat_completion.choices[0].message.content
        print(messages, answer)
        return answer

    def get_system_prompt(self, user_id):
        query = Query()
        if self.system_prompts_table.contains(query.user_id == user_id):
            return self.system_prompts_table.get(query.user_id == user_id)["prompt"]
        return DEFAULT_SYSTEM_PROMPT

    def set_system_prompt(self, user_id: int, text: str):
        query = Query()
        if self.system_prompts_table.contains(query.user_id == user_id):
            self.system_prompts_table.update(ops.set("prompt", text), query.user_id == user_id)
        else:
            self.system_prompts_table.insert({"prompt": text, "user_id": user_id})

    async def start(self, message: Message):
        user_id = message.from_user.id
        self.create_conv_id(user_id)
        await message.reply(f"Привет {message.from_user.first_name}! Как тебе помочь?")

    async def set_system(self, message: Message):
        user_id = message.from_user.id
        text = message.text.replace("/set_system", "").strip()
        self.set_system_prompt(user_id, text)
        await message.reply(f"Новый системный промпт задан:\n\n{text}")

    async def get_system(self, message: Message):
        user_id = message.from_user.id
        prompt = self.get_system_prompt(user_id)
        if prompt.strip():
            await message.reply(prompt)
        else:
            await message.reply("Системный промпт пуст")

    async def reset_system(self, message: Message):
        user_id = message.from_user.id
        self.set_system_prompt(user_id, DEFAULT_SYSTEM_PROMPT)
        await message.reply("Системный промпт сброшен!")

    async def reset(self, message: Message):
        user_id = message.from_user.id
        self.create_conv_id(user_id)
        await message.reply("История сброшена!")

    async def history(self, message: Message):
        user_id = message.from_user.id
        conv_id = self.get_current_conv_id(user_id)
        history = self.fetch_conversation(conv_id)
        history = json.dumps(history, ensure_ascii=False)
        history = history[:3000] + "... truncated"
        await message.reply(history)

    async def generate(self, message: Message):
        user_id = message.from_user.id
        last_message = message.text
        conv_id = self.get_current_conv_id(user_id)
        history = self.fetch_conversation(conv_id)
        self.messages_table.insert({
            "role": "user",
            "content": last_message,
            "conv_id": conv_id,
            "timestamp": self.get_current_ts()
        })
        system_prompt = self.get_system_prompt(user_id)
        placeholder = await message.answer("💬")
        answer = await self.query_api(
            history=history,
            last_message=last_message,
            system_prompt=system_prompt
        )
        builder = InlineKeyboardBuilder()
        builder.add(InlineKeyboardButton(
            text="👍",
            callback_data="like"
        ))
        builder.add(InlineKeyboardButton(
            text="👎",
            callback_data="dislike"
        ))
        message = await placeholder.edit_text(answer, reply_markup=builder.as_markup())
        self.messages_table.insert({
            "role": "assistant",
            "content": answer,
            "conv_id": conv_id,
            "timestamp": self.get_current_ts(),
            "message_id": message.message_id
        })

    async def save_like(self, callback: CallbackQuery):
        user_id = callback.from_user.id
        message_id = callback.message.message_id
        self.likes_table.insert({
            "user_id": user_id,
            "message_id": message_id,
            "feedback": "like"
        })
        await self.bot.edit_message_reply_markup(
            chat_id=callback.message.chat.id,
            message_id=message_id,
            reply_markup=None
        )

    async def save_dislike(self, callback: CallbackQuery):
        user_id = callback.from_user.id
        message_id = callback.message.message_id
        self.likes_table.insert({
            "user_id": user_id,
            "message_id": message_id,
            "feedback": "dislike"
        })
        await self.bot.edit_message_reply_markup(
            chat_id=callback.message.chat.id,
            message_id=message_id,
            reply_markup=None
        )


def main(
    bot_token: str,
    api_key: str,
    base_url: str,
    db_path: str,
    model_name: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_tokens: int = 1536,
    history_max_tokens: int = 6144
) -> None:
    bot = LlmBot(
        bot_token=bot_token,
        api_key=api_key,
        base_url=base_url,
        db_path=db_path,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        history_max_tokens=history_max_tokens
    )
    asyncio.run(bot.start_polling())


if __name__ == "__main__":
    fire.Fire(main)
