import asyncio
import os
import json
import traceback
import base64

import fire
import tiktoken
from aiogram import Bot, Dispatcher
from aiogram import F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import Message, InlineKeyboardButton, CallbackQuery
from aiogram.utils.keyboard import InlineKeyboardBuilder
from openai import AsyncOpenAI
from transformers import AutoTokenizer
from database import Database

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."
DEFAULT_MODEL = "gpt-4o"


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
        client_config_path: str,
        db_path: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        history_max_tokens: int,
        chunk_size: int,
        user_message_limit: int
    ):
        # Клиент
        with open(client_config_path) as r:
            client_config = json.load(r)
        self.clients = dict()
        self.model_names = dict()
        self.can_handle_images = dict()
        for model_name, config in client_config.items():
            self.model_names[model_name] = config.pop("model_name")
            self.can_handle_images[model_name] = config.pop("can_handle_images", False)
            self.clients[model_name] = AsyncOpenAI(**config)
        assert self.clients
        assert self.model_names

        self.user_message_limit = user_message_limit

        # Параметры
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.history_max_tokens = history_max_tokens

        # База
        self.db = Database(db_path)

        # Клавиатуры
        self.inline_models_list_kb = InlineKeyboardBuilder()
        for model_id in self.clients.keys():
            self.inline_models_list_kb.add(InlineKeyboardButton(text=model_id, callback_data=f"setmodel:{model_id}"))

        # Бот
        self.chunk_size = chunk_size
        self.bot = Bot(token=bot_token, default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN))
        self.dp = Dispatcher()
        self.dp.message.register(self.start, Command("start"))
        self.dp.message.register(self.reset, Command("reset"))
        self.dp.message.register(self.history, Command("history"))
        self.dp.message.register(self.set_system, Command("setsystem"))
        self.dp.message.register(self.get_system, Command("getsystem"))
        self.dp.message.register(self.reset_system, Command("resetsystem"))
        self.dp.message.register(self.set_model, Command("setmodel"))
        self.dp.message.register(self.get_model, Command("getmodel"))
        self.dp.message.register(self.get_count, Command("getcount"))
        self.dp.message.register(self.generate)
        self.dp.callback_query.register(self.save_feedback, F.data.startswith("feedback:"))
        self.dp.callback_query.register(self.set_model_button_handler, F.data.startswith("setmodel:"))

    async def start_polling(self):
        await self.dp.start_polling(self.bot)

    async def start(self, message: Message):
        user_id = message.from_user.id
        self.db.create_conv_id(user_id)
        await message.reply("Привет! Как тебе помочь?")

    async def get_count(self, message: Message) -> int:
        user_id = message.from_user.id
        count = self.db.count_user_messages(user_id)
        await message.reply("Осталось запросов: {}".format(self.user_message_limit - count))

    async def set_system(self, message: Message):
        user_id = message.from_user.id
        text = message.text.replace("/setsystem", "").strip()
        self.db.set_system_prompt(user_id, text)
        self.db.create_conv_id(user_id)
        await message.reply(f"Новый системный промпт задан:\n\n{text}")

    async def get_system(self, message: Message):
        user_id = message.from_user.id
        prompt = self.db.get_system_prompt(user_id)
        if prompt.strip():
            await message.reply(prompt)
        else:
            await message.reply("Системный промпт пуст")

    async def reset_system(self, message: Message):
        user_id = message.from_user.id
        self.db.set_system_prompt(user_id, DEFAULT_SYSTEM_PROMPT)
        self.db.create_conv_id(user_id)
        await message.reply("Системный промпт сброшен!")

    async def get_model(self, message: Message):
        user_id = message.from_user.id
        model = self.db.get_current_model(user_id)
        await message.reply(model)

    async def set_model(self, message: Message):
        await message.reply("Выберите модель:", reply_markup=self.inline_models_list_kb.as_markup())

    async def reset(self, message: Message):
        user_id = message.from_user.id
        self.db.create_conv_id(user_id)
        await message.reply("История сообщений сброшена!")

    async def history(self, message: Message):
        user_id = message.from_user.id
        conv_id = self.db.get_current_conv_id(user_id)
        history = self.db.fetch_conversation(conv_id)
        for m in history:
            if not isinstance(m["content"], str):
                m["content"] = "Not text"
        history = json.dumps(history, ensure_ascii=False)
        if len(history) > self.chunk_size:
            history = history[:self.chunk_size] + "... truncated"
        await message.reply(history, parse_mode=None)

    async def generate(self, message: Message):
        user_id = message.from_user.id
        model = self.db.get_current_model(user_id)
        if model == "gpt-4o":
            count = self.db.count_user_messages(user_id)
            print(user_id, count)
            if count > self.user_message_limit:
                print(user_id, "limit")
                await message.answer("Вы превысили лимит запросов по gpt-4o, переключите модель на другую с помощью /setmodel")
                return

        if model not in self.clients:
            await message.answer("Выбранная модель больше не поддерживается, переключите на другую с помощью /setmodel")
            return

        conv_id = self.db.get_current_conv_id(user_id)
        history = self.db.fetch_conversation(conv_id)
        system_prompt = self.db.get_system_prompt(user_id)

        content = await self._build_content(message)
        if not isinstance(content, str) and not self.can_handle_images[model]:
            await message.answer("Выбранная модель не может обработать ваше сообщение")
            return
        if content is None:
            await message.answer("Такой тип сообщений (ещё) не поддерживается")
            return

        self.db.save_user_message(content, conv_id=conv_id)
        placeholder = await message.answer("💬")

        try:
            answer = await self._query_api(
                model=model,
                history=history,
                user_content=content,
                system_prompt=system_prompt
            )

            builder = InlineKeyboardBuilder()
            builder.add(InlineKeyboardButton(
                text="👍",
                callback_data="feedback:like"
            ))
            builder.add(InlineKeyboardButton(
                text="👎",
                callback_data="feedback:dislike"
            ))
            markup = builder.as_markup()

            chunk_size = self.chunk_size
            answer_parts = [answer[i:i + chunk_size] for i in range(0, len(answer), chunk_size)]

            new_message = await placeholder.edit_text(answer_parts[0], parse_mode=None)
            for part in answer_parts[1:]:
                new_message = await message.answer(part, parse_mode=None)
            new_message = await new_message.edit_text(answer_parts[-1], parse_mode=None, reply_markup=markup)

            self.db.save_assistant_message(
                content=answer,
                conv_id=conv_id,
                message_id=new_message.message_id,
                model=model,
                system_prompt=system_prompt
            )

        except Exception:
            traceback.print_exc()
            await placeholder.edit_text("Что-то пошло не так, ответ от Сайги не получен или не смог отобразиться.")

    async def save_feedback(self, callback: CallbackQuery):
        user_id = callback.from_user.id
        message_id = callback.message.message_id
        feedback = callback.data.split(":")[1]
        self.db.save_feedback(feedback, user_id=user_id, message_id=message_id)
        await self.bot.edit_message_reply_markup(
            chat_id=callback.message.chat.id,
            message_id=message_id,
            reply_markup=None
        )

    async def set_model_button_handler(self, callback: CallbackQuery):
        user_id = callback.from_user.id
        model_name = callback.data.split(":")[1]
        assert model_name in self.clients
        if model_name in self.clients:
            self.db.set_current_model(user_id, model_name)
            self.db.create_conv_id(user_id)
            await self.bot.send_message(chat_id=user_id, text=f"Новая модель задана:\n\n{model_name}")
        else:
            model_list = list(self.clients.keys())
            await self.bot.send_message(chat_id=user_id, text=f"Некорректное имя модели. Выберите из: {model_list}")

    def _count_tokens(self, messages, model):
        if "api.openai.com" in str(self.clients[model].base_url):
            encoding = tiktoken.encoding_for_model(self.model_names[model])
            tokens_count = 0
            for m in messages:
                if isinstance(m["content"], str):
                    tokens_count += len(encoding.encode(m["content"]))
                else:
                    tokens_count += 1000
        else:
            tokenizer = Tokenizer.get(self.model_names[model])
            tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            tokens_count = len(tokens)
        return tokens_count

    @staticmethod
    def _merge_messages(messages):
        new_messages = []
        prev_role = None
        for m in messages:
            if m["content"] is None:
                continue
            if m["role"] == prev_role:
                is_current_str = isinstance(m["content"], str)
                is_prev_str = isinstance(new_messages[-1]["content"], str)
                if is_current_str and is_prev_str:
                    new_messages[-1]["content"] += "\n" + m["content"]
                    continue
            prev_role = m["role"]
            new_messages.append(m)
        return new_messages

    def _crop_content(self, content):
        if isinstance(content, str):
            return content.replace("\n", " ")[:40]
        return "Not text"

    async def _query_api(self, model, history, user_content, system_prompt: str):
        messages = history + [{"role": "user", "content": user_content}]
        messages = self._merge_messages(messages)
        assert messages

        tokens_count = self._count_tokens(messages, model=model)
        while tokens_count > self.history_max_tokens and len(messages) >= 3:
            messages = messages[2:]
            tokens_count = self._count_tokens(messages, model=model)

        assert messages
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": system_prompt})

        print(model, "####", len(messages), "####", self._crop_content(messages[-1]["content"]))
        chat_completion = await self.clients[model].chat.completions.create(
            model=self.model_names[model],
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens
        )
        answer = chat_completion.choices[0].message.content
        print(
            model, "####",
            len(messages), "####",
            self._crop_content(messages[-1]["content"]), "####",
            self._crop_content(answer)
        )
        return answer

    async def _build_content(self, message: Message):
        content_type = message.content_type
        if content_type == "text":
            return message.text

        photo = None
        photo_ext = (".jpg", "jpeg", ".png", ".webp", ".gif")
        if content_type == "photo":
            document = message.photo[-1]
            file_info = await self.bot.get_file(document.file_id)
            photo = file_info.file_path
        elif content_type == "document":
            document = message.document
            file_info = await self.bot.get_file(document.file_id)
            file_path = file_info.file_path
            if "." + file_path.split(".")[-1].lower() in photo_ext:
                photo = file_path

        if photo:
            file_stream = await self.bot.download_file(photo)
            assert file_stream
            file_stream.seek(0)
            base64_image = base64.b64encode(file_stream.read()).decode("utf-8")
            assert base64_image
            content = []
            if message.caption:
                content.append({
                    "type": "text",
                    "text": message.caption
                })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
            return content

        return None


def main(
    bot_token: str,
    client_config_path: str,
    db_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_tokens: int = 1536,
    history_max_tokens: int = 6144,
    chunk_size: int = 3500,
    user_message_limit: int = 100
) -> None:
    bot = LlmBot(
        bot_token=bot_token,
        client_config_path=client_config_path,
        db_path=db_path,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        history_max_tokens=history_max_tokens,
        chunk_size=chunk_size,
        user_message_limit=user_message_limit
    )
    asyncio.run(bot.start_polling())


if __name__ == "__main__":
    fire.Fire(main)
