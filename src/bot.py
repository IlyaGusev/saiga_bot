import asyncio
import io
import os
import json
import copy
import traceback
import base64
import textwrap
from functools import wraps

import requests
import fire
import tiktoken
from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode, ChatMemberStatus
from aiogram.filters import Command
from aiogram.types import Message, InlineKeyboardButton, CallbackQuery, BufferedInputFile
from aiogram.utils.keyboard import InlineKeyboardBuilder
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from src.tools import Tool
from src.database import Database

os.environ["TOKENIZERS_PARALLELISM"] = "false"


DEFAULT_MESSAGE_COUNT_LIMIT = {
    "standard": {"limit": 1000, "interval": 86400},
    "subscribed": {"limit": 1000, "interval": 86400},
}
TEMPERATURE_RANGE = (0.0, 0.5, 0.8, 1.0, 1.2)
TOP_P_RANGE = (0.8, 0.9, 0.95, 0.98, 1.0)
DALLE_DAILY_LIMIT = 5
START_TEMPLATE = """
Привет! Я Сайга, бот с разными языковыми моделями.
Текущая модель: {model}
Осталось сообщений: {message_count}

Доступные команды:
/reset - очистить историю сообщений с ботом
/setmodel - выбрать модель
/getmodel - узнать текущую модель
/setcharacter - задать персонажа
/setsystem ... - задать системный промпт
/getsystem - узнать текущий системный промпт
/resetsystem - сбросить системный промпт к стандартному
/setshortname ... - задать имя бота
/getshortname - узнать имя ботя
/getcount - узнать текущий лимит сообщений
/getparams - узнать текущие параметры генерации
/settemperature - задать температуру

Исходники: https://github.com/IlyaGusev/saiga_bot
Модель по умолчанию: https://huggingface.co/IlyaGusev/saiga_llama3_8b

По всем вопросам писать @YallenGusev
"""

IMAGE_PLACEHOLDER = "<image_placeholder>"
BUY_TITLE = "*Покупка подписки на неделю*"
BUT_DESCRIPTION = """Лимиты общения с моделями станут такими:
{sub_limits}

Подписка будет действовать ровно 7 дней.

Цена подписки: *500 рублей*
"""


class Tokenizer:
    tokenizers = dict()

    @classmethod
    def get(cls, model_name: str):
        if model_name not in cls.tokenizers:
            cls.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
        return cls.tokenizers[model_name]


def check_admin(func):
    @wraps(func)
    async def wrapped(self, obj, *args, **kwargs):
        if isinstance(obj, CallbackQuery):
            chat_id = obj.message.chat.id
            user_id = obj.from_user.id
            user = obj.from_user
        elif isinstance(obj, Message):
            chat_id = obj.chat.id
            user_id = obj.from_user.id
            user = obj.from_user
        else:
            return await func(self, obj, *args, **kwargs)

        if chat_id != user_id:
            user_name = self._get_user_name(user)
            is_admin = await self._is_admin(user_id=user_id, chat_id=chat_id)
            if not is_admin:
                await self.bot.send_message(
                    chat_id=chat_id,
                    text=f"{user_name}, только админы могут это делать",
                )
                return
        return await func(self, obj, *args, **kwargs)

    return wrapped


class LlmBot:
    def __init__(
        self,
        bot_token: str,
        client_config_path: str,
        db_path: str,
        history_max_tokens: int,
        chunk_size: int,
        characters_path: str,
        tools_config_path: str,
    ):
        # Клиент
        with open(client_config_path) as r:
            client_config = json.load(r)
        self.clients = dict()
        self.model_names = dict()
        self.can_handle_images = dict()
        self.can_handle_tools = dict()
        self.default_prompts = dict()
        self.default_params = dict()
        self.limits = dict()
        for model_name, config in client_config.items():
            self.model_names[model_name] = config.pop("model_name")
            self.can_handle_images[model_name] = config.pop("can_handle_images", False)
            self.can_handle_tools[model_name] = config.pop("can_handle_tools", False)
            self.default_prompts[model_name] = config.pop("system_prompt", "")
            if "params" in config:
                self.default_params[model_name] = config.pop("params")
            self.limits[model_name] = config.pop("message_count_limit", DEFAULT_MESSAGE_COUNT_LIMIT)
            assert "standard" in self.limits[model_name]
            assert "subscribed" in self.limits[model_name]
            self.clients[model_name] = AsyncOpenAI(**config)
        assert self.clients
        assert self.model_names
        assert self.default_prompts

        # Инструменты
        self.tools = dict()
        if tools_config_path:
            assert os.path.exists(tools_config_path)
            with open(tools_config_path) as r:
                tools_config = json.load(r)
                for tool_name, tool_config in tools_config.items():
                    self.tools[tool_name] = Tool.by_name(tool_name)(**tool_config)

        # Персонажи
        self.characters = dict()
        if characters_path and os.path.exists(characters_path):
            with open(characters_path) as r:
                self.characters = json.load(r)

        # Параметры
        self.history_max_tokens = history_max_tokens
        self.chunk_size = chunk_size

        # База
        self.db = Database(db_path)

        # Клавиатуры
        self.models_kb = InlineKeyboardBuilder()
        for model_id in self.clients.keys():
            self.models_kb.row(InlineKeyboardButton(text=model_id, callback_data=f"setmodel:{model_id}"))
        self.models_kb.adjust(2)

        self.characters_kb = InlineKeyboardBuilder()
        for char_id in self.characters.keys():
            self.characters_kb.row(InlineKeyboardButton(text=char_id, callback_data=f"setcharacter:{char_id}"))
        self.characters_kb.adjust(2)

        self.likes_kb = InlineKeyboardBuilder()
        self.likes_kb.add(InlineKeyboardButton(text="👍", callback_data="feedback:like"))
        self.likes_kb.add(InlineKeyboardButton(text="👎", callback_data="feedback:dislike"))

        self.temperature_kb = InlineKeyboardBuilder()
        for value in TEMPERATURE_RANGE:
            self.temperature_kb.add(InlineKeyboardButton(text=str(value), callback_data=f"settemperature:{value}"))

        self.top_p_kb = InlineKeyboardBuilder()
        for value in TOP_P_RANGE:
            self.top_p_kb.add(InlineKeyboardButton(text=str(value), callback_data=f"settopp:{value}"))

        self.buy_kb = InlineKeyboardBuilder()
        self.buy_kb.add(InlineKeyboardButton(text="Купить", callback_data="buy:0"))

        # Бот
        self.bot = Bot(token=bot_token, default=DefaultBotProperties(parse_mode=None))
        self.bot_info = None
        self.dp = Dispatcher()
        self.dp.message.register(self.start, Command("start"))
        self.dp.message.register(self.start, Command("help"))
        self.dp.message.register(self.reset, Command("reset"))
        self.dp.message.register(self.set_system, Command("setsystem"))
        self.dp.message.register(self.get_system, Command("getsystem"))
        self.dp.message.register(self.reset_system, Command("resetsystem"))
        self.dp.message.register(self.set_model, Command("setmodel"))
        self.dp.message.register(self.get_model, Command("getmodel"))
        self.dp.message.register(self.set_short_name, Command("setshortname"))
        self.dp.message.register(self.get_short_name, Command("getshortname"))
        self.dp.message.register(self.set_character, Command("setcharacter"))
        self.dp.message.register(self.get_count, Command("getcount"))
        self.dp.message.register(self.get_params, Command("getparams"))
        self.dp.message.register(self.set_temperature, Command("settemperature"))
        self.dp.message.register(self.set_top_p, Command("settopp"))
        self.dp.message.register(self.sub_info, Command("subinfo"))
        self.dp.message.register(self.sub_buy, Command("subbuy"))
        self.dp.message.register(self.history, Command("history"))
        self.dp.message.register(self.generate)

        self.dp.callback_query.register(self.save_feedback, F.data.startswith("feedback:"))
        self.dp.callback_query.register(self.set_model_button_handler, F.data.startswith("setmodel:"))
        self.dp.callback_query.register(self.set_character_button_handler, F.data.startswith("setcharacter:"))
        self.dp.callback_query.register(self.set_temperature_button_handler, F.data.startswith("settemperature:"))
        self.dp.callback_query.register(self.set_top_p_button_handler, F.data.startswith("settopp:"))
        self.dp.callback_query.register(self.sub_buy_proceed, F.data.startswith("buy:"))

    async def start_polling(self):
        self.bot_info = await self.bot.get_me()
        await self.dp.start_polling(self.bot)

    async def start(self, message: Message):
        user_id = message.from_user.id
        chat_id = message.chat.id
        self.db.create_conv_id(chat_id)
        model = self.db.get_current_model(chat_id)
        remaining_count = self.count_remaining_messages(user_id=user_id, model=model)
        content = START_TEMPLATE.format(model=model, message_count=remaining_count)
        await message.reply(content)

    #
    # История сообщений
    #

    async def reset(self, message: Message):
        chat_id = message.chat.id
        self.db.create_conv_id(chat_id)
        await message.reply("История сообщений сброшена!")

    async def history(self, message: Message):
        chat_id = message.chat.id
        is_chat = chat_id != message.from_user.id
        conv_id = self.db.get_current_conv_id(chat_id)
        history = self.db.fetch_conversation(conv_id)
        if not history:
            await message.reply("Нет истории!")
            return
        history = self._replace_images(history)
        model = self.db.get_current_model(chat_id)
        history = self._prepare_history(history, model=model, is_chat=is_chat)
        history = json.dumps(history, ensure_ascii=False)
        history = self._truncate_text(history)
        await message.reply(history)

    async def _save_chat_message(self, message: Message):
        chat_id = message.chat.id
        user_id = message.from_user.id
        user_name = self._get_user_name(message.from_user)
        content = await self._build_content(message)
        if content is None:
            return
        conv_id = self.db.get_current_conv_id(chat_id)
        self.db.save_user_message(content, conv_id=conv_id, user_id=user_id, user_name=user_name)

    #
    # Выбор модели
    #

    @check_admin
    async def set_model(self, message: Message):
        await message.reply("Выберите модель:", reply_markup=self.models_kb.as_markup())

    @check_admin
    async def set_model_button_handler(self, callback: CallbackQuery):
        chat_id = callback.message.chat.id
        model_name = callback.data.split(":")[1]
        assert model_name in self.clients
        self.db.set_current_model(chat_id, model_name)
        self.db.create_conv_id(chat_id)
        await callback.message.edit_text(f"Новая модель задана:\n\n{model_name}")

    async def get_model(self, message: Message):
        chat_id = message.chat.id
        model = self.db.get_current_model(chat_id)
        await message.reply(model)

    #
    # Системный промпт
    #

    @check_admin
    async def set_system(self, message: Message):
        chat_id = message.chat.id
        text = message.text.replace("/setsystem", "").strip()
        text = text.replace("@{}".format(self.bot_info.username), "").strip()
        self.db.set_system_prompt(chat_id, text)
        self.db.create_conv_id(chat_id)
        text = self._truncate_text(text)
        await message.reply(f"Новый системный промпт задан:\n\n{text}")

    async def get_system(self, message: Message):
        chat_id = message.chat.id
        prompt = self.db.get_system_prompt(chat_id, self.default_prompts)
        if not prompt.strip():
            await message.reply("Системный промпт пуст")
            return
        await message.reply(prompt)

    async def reset_system(self, message: Message):
        chat_id = message.chat.id
        model = self.db.get_current_model(chat_id)
        self.db.set_system_prompt(chat_id, self.default_prompts.get(model, ""))
        self.db.create_conv_id(chat_id)
        await message.reply("Системный промпт сброшен!")

    #
    # Именование модели
    #

    @check_admin
    async def set_short_name(self, message: Message):
        chat_id = message.chat.id
        text = message.text.replace("/setshortname", "").strip()
        text = text.replace("@{}".format(self.bot_info.username), "").strip()
        if not text:
            await message.reply("Короткое имя не может быть пустым. Напишите имя в одном сообщении с командой.")
            return
        self.db.set_short_name(chat_id, text)
        self.db.create_conv_id(chat_id)
        await message.reply(f"Новое короткое имя задано:\n\n{text}")

    async def get_short_name(self, message: Message):
        chat_id = message.chat.id
        name = self.db.get_short_name(chat_id)
        await message.reply(f"Короткое имя бота: {name}")

    #
    # Персонажи
    #

    @check_admin
    async def set_character(self, message: Message):
        await message.reply("Выберите персонажа:", reply_markup=self.characters_kb.as_markup())

    @check_admin
    async def set_character_button_handler(self, callback: CallbackQuery):
        chat_id = callback.message.chat.id
        char_name = callback.data.split(":")[1]
        assert char_name in self.characters
        character = self.characters[char_name]
        system_prompt = character["system_prompt"]
        self.db.set_system_prompt(chat_id, system_prompt)
        short_name = character["short_name"]
        self.db.set_short_name(chat_id, short_name)
        self.db.create_conv_id(chat_id)
        await callback.message.edit_text(
            f"Новый персонаж задан:\n\n{system_prompt}\n\nМожно обращаться так: '{short_name}'"
        )

    #
    # Лимиты
    #

    def count_remaining_messages(self, user_id: int, model: str):
        is_subscribed = self.db.is_subscribed_user(user_id)
        mode = "standard" if not is_subscribed else "subscribed"
        limit = self.limits[model][mode]["limit"]
        interval = self.limits[model][mode]["interval"]
        count = self.db.count_user_messages(user_id, model, interval)
        remaining_count = limit - count
        return max(0, remaining_count)

    async def get_count(self, message: Message) -> int:
        user_id = message.from_user.id
        chat_id = message.chat.id
        model = self.db.get_current_model(chat_id)
        remaining_count = self.count_remaining_messages(user_id=user_id, model=model)
        await message.reply("Осталось запросов к {}: {}".format(model, remaining_count))

    async def sub_info(self, message: Message):
        user_id = message.from_user.id
        remaining_seconds = self.db.get_subscription_info(user_id)
        text = "Подписка неактивна"
        if remaining_seconds > 0:
            text = f"Подписка активирована! Осталось {remaining_seconds//3600}ч"
        await message.reply(text)

    async def sub_buy(self, message: Message):
        user_id = message.from_user.id
        remaining_seconds = self.db.get_subscription_info(user_id)
        if remaining_seconds > 0:
            await message.reply(f"У вас уже есть подписка! Она закончится через {remaining_seconds//3600}ч")
            return
        template = "- *{model}*: {count} сообщений каждые {hours} часа"
        sub_limits = [
            template.format(
                model=model, count=limit["subscribed"]["limit"], hours=limit["subscribed"]["interval"] // 3600
            )
            for model, limit in self.limits.items()
        ]
        description = BUT_DESCRIPTION.format(sub_limits="\n".join(sub_limits))
        text = BUY_TITLE + "\n\n" + description
        await message.reply(text, parse_mode=ParseMode.MARKDOWN, reply_markup=self.buy_kb.as_markup())

    async def sub_buy_proceed(self, callback: CallbackQuery):
        await callback.message.reply("Ссылка для оплаты: <ссылка>")

    #
    # Параметры генерации
    #

    @check_admin
    async def set_temperature(self, message: Message):
        await message.reply("Выберите температуру:", reply_markup=self.temperature_kb.as_markup())

    @check_admin
    async def set_temperature_button_handler(self, callback: CallbackQuery):
        chat_id = callback.message.chat.id
        temperature = float(callback.data.split(":")[1])
        self.db.set_parameters(chat_id, self.default_params, temperature=temperature)
        await callback.message.edit_text(f"Новая температура задана:\n\n{temperature}")

    @check_admin
    async def set_top_p(self, message: Message):
        await message.reply("Выберите top-p:", reply_markup=self.top_p_kb.as_markup())

    @check_admin
    async def set_top_p_button_handler(self, callback: CallbackQuery):
        chat_id = callback.message.chat.id
        top_p = float(callback.data.split(":")[1])
        self.db.set_parameters(chat_id, self.default_params, top_p=top_p)
        await callback.message.edit_text(f"Новое top-p задано:\n\n{top_p}")

    async def get_params(self, message: Message):
        chat_id = message.chat.id
        params = self.db.get_parameters(chat_id, self.default_params)
        await message.reply(f"Текущие параметры генерации: {json.dumps(params)}")

    #
    # Инструменты
    #

    async def _check_tools(self, messages, model: str):
        messages = copy.deepcopy(messages)
        messages = self._replace_images(messages)
        tools = [t.get_specification() for t in self.tools.values()]
        chat_completion = await self.clients[model].chat.completions.create(
            model=self.model_names[model], messages=messages, tools=tools, tool_choice="auto", max_tokens=2048
        )
        response_message = chat_completion.choices[0].message
        return response_message

    async def _call_dalle(self, conv_id: str, user_id: int, chat_id: int, placeholder: Message, **kwargs):
        dalle_count = self.db.count_generated_images(user_id, 86400)
        is_dalle_remaining = DALLE_DAILY_LIMIT - dalle_count > 0
        if not is_dalle_remaining:
            await placeholder.edit_text("Лимит по генерации картинок исчерпан, восстановится через 24 часа")
            return

        await placeholder.edit_text(f"Генерирую картинку по промпту: {kwargs['prompt_russian']}")
        function_response = await self.tools["dalle"](**kwargs)
        base64_image = function_response[1]["image_url"]["url"].replace("data:image/jpeg;base64,", "")
        image_data = base64.b64decode(base64_image)
        input_file = BufferedInputFile(image_data, filename="image.jpeg")
        new_message = await self.bot.send_photo(
            chat_id=chat_id, photo=input_file, reply_to_message_id=placeholder.message_id
        )
        self.db.save_assistant_message(
            content=function_response,
            conv_id=conv_id,
            message_id=new_message.message_id,
            model="dalle",
            reply_user_id=user_id,
        )

    async def _call_tools(self, history, model: str, conv_id: str, user_id: int, chat_id: int, placeholder: Message):
        response_message = await self._check_tools(history, model=model)
        tool_calls = response_message.tool_calls
        if not tool_calls:
            return history

        tool_calls_dict = [c.to_dict() for c in tool_calls]
        response_message = {"content": None, "role": "assistant", "tool_calls": tool_calls_dict}
        history.append(response_message)
        self.db.save_tool_calls_message(conv_id=conv_id, model=model, tool_calls=tool_calls_dict)

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = self.tools[function_name]
            print(function_name, "call", tool_call.function.arguments)

            function_response = None
            try:
                function_args = json.loads(tool_call.function.arguments)
            except json.decoder.JSONDecodeError:
                function_response = "No response from the tool, try again"
                print(f"Bad tool answer: {tool_call.function.arguments}")

            if function_response is None and function_name == "dalle":
                await self._call_dalle(
                    chat_id=chat_id, user_id=user_id, conv_id=conv_id, placeholder=placeholder, **function_args
                )
                return None

            if function_response is None:
                try:
                    function_response = await function_to_call(**function_args)
                except Exception as e:
                    function_response = f"Ошибка при вызове инструмента: {str(e)}"

            history.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )
            self.db.save_tool_answer_message(
                tool_call_id=tool_call.id,
                content=function_response,
                conv_id=conv_id,
                model=model,
                name=function_name,
            )
        return history

    #
    # Генерация
    #

    async def generate(self, message: Message):
        user_id = message.from_user.id
        user_name = self._get_user_name(message.from_user)
        chat_id = user_id
        is_chat = False
        if message.chat.type in ("group", "supergroup"):
            chat_id = message.chat.id
            is_chat = True
            is_reply = message.reply_to_message and message.reply_to_message.from_user.id == self.bot_info.id
            bot_short_name = self.db.get_short_name(chat_id)
            bot_names = ["@" + self.bot_info.username, bot_short_name]
            is_explicit = message.text and any(bot_name in message.text for bot_name in bot_names)
            if not is_reply and not is_explicit:
                await self._save_chat_message(message)
                return

        model = self.db.get_current_model(chat_id)
        if model not in self.clients:
            await message.reply("Выбранная модель больше не поддерживается, переключите на другую с помощью /setmodel")
            return

        remaining_count = self.count_remaining_messages(user_id=user_id, model=model)
        print(user_id, model, remaining_count)
        if remaining_count <= 0:
            await message.reply(
                f"Превышен дневной лимит запросов по {model}, подождите или переключите модель с /setmodel"
            )
            return

        params = self.db.get_parameters(chat_id, self.default_params)
        if "claude" in model and params["temperature"] > 1.0:
            await message.reply("Claude не поддерживает температуру выше 1, задайте новую с помощью /settemperature")
            return

        conv_id = self.db.get_current_conv_id(chat_id)
        history = self.db.fetch_conversation(conv_id)
        system_prompt = self.db.get_system_prompt(chat_id, self.default_prompts)

        content = await self._build_content(message)
        if not isinstance(content, str) and not self.can_handle_images[model]:
            await message.reply("Выбранная модель не может обработать ваше сообщение")
            return
        if content is None:
            await message.reply("Такой тип сообщений (ещё) не поддерживается")
            return

        self.db.save_user_message(content, conv_id=conv_id, user_id=user_id, user_name=user_name)

        history = history + [{"role": "user", "content": content, "user_name": user_name}]
        history = self._prepare_history(history, model=model, is_chat=is_chat)

        placeholder = await message.reply("💬")

        try:
            if self.can_handle_tools[model] and self.tools:
                history = await self._call_tools(
                    history=history,
                    model=model,
                    user_id=user_id,
                    conv_id=conv_id,
                    chat_id=chat_id,
                    placeholder=placeholder,
                )
                if history is None:
                    return

            history = self._fix_image_roles(history)
            history = self._fix_broken_tool_calls(history)
            answer = await self._query_api(model=model, messages=history, system_prompt=system_prompt, **params)

            chunk_size = self.chunk_size
            answer_parts = [answer[i : i + chunk_size] for i in range(0, len(answer), chunk_size)]
            new_message = await placeholder.edit_text(answer_parts[0])
            for part in answer_parts[1:]:
                new_message = await message.reply(part)

            markup = self.likes_kb.as_markup()
            new_message = await new_message.edit_text(
                answer_parts[-1],
                reply_markup=markup,
            )
            self.db.save_assistant_message(
                content=answer,
                conv_id=conv_id,
                message_id=new_message.message_id,
                model=model,
                system_prompt=system_prompt,
                reply_user_id=user_id,
            )

        except Exception:
            traceback.print_exc()
            await placeholder.edit_text("Что-то пошло не так, ответ от Сайги не получен или не смог отобразиться.")

    async def _query_api(self, model, messages, system_prompt: str, **kwargs):
        assert messages
        if messages[0]["role"] != "system" and system_prompt.strip():
            messages.insert(0, {"role": "system", "content": system_prompt})

        print(
            model,
            "####",
            len(messages),
            "####",
            self._crop_content(messages[-1]["content"]),
        )
        chat_completion = await self.clients[model].chat.completions.create(
            model=self.model_names[model], messages=messages, **kwargs
        )
        answer = chat_completion.choices[0].message.content
        print(
            model,
            "####",
            len(messages),
            "####",
            self._crop_content(messages[-1]["content"]),
            "####",
            self._crop_content(answer),
        )
        return answer

    async def _build_content(self, message: Message):
        content_type = message.content_type
        if content_type == "text":
            text = message.text
            chat_id = message.chat.id
            bot_short_name = self.db.get_short_name(chat_id)
            text = text.replace("@" + self.bot_info.username, bot_short_name).strip()
            return text

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
                content.append({"type": "text", "text": message.caption})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                }
            )
            return content

        return None

    #
    # Служебные методы
    #

    async def save_feedback(self, callback: CallbackQuery):
        user_id = callback.from_user.id
        message_id = callback.message.message_id
        feedback = callback.data.split(":")[1]
        self.db.save_feedback(feedback, user_id=user_id, message_id=message_id)
        await self.bot.edit_message_reply_markup(
            chat_id=callback.message.chat.id, message_id=message_id, reply_markup=None
        )

    def _count_tokens(self, messages, model):
        url = str(self.clients[model].base_url)
        tokens_count = 0

        if "api.openai.com" in url:
            encoding = tiktoken.encoding_for_model(self.model_names[model])
            for m in messages:
                if isinstance(m["content"], str):
                    tokens_count += len(encoding.encode(m["content"]))
                elif self._is_image_content(m["content"]):
                    tokens_count += 1000
            return tokens_count

        if "anthropic" in url:
            for m in messages:
                if isinstance(m["content"], str):
                    tokens_count += len(m["content"]) // 2
            return tokens_count

        tokenizer = Tokenizer.get(self.model_names[model])
        tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        tokens_count = len(tokens)
        return tokens_count

    async def _is_admin(self, user_id, chat_id):
        chat_member = await self.bot.get_chat_member(chat_id, user_id)
        return chat_member.status in [
            ChatMemberStatus.ADMINISTRATOR,
            ChatMemberStatus.CREATOR,
        ]

    @staticmethod
    def _merge_messages(messages):
        new_messages = []
        prev_role = None
        for m in messages:
            content = m["content"]
            role = m["role"]
            if role == prev_role:
                is_current_str = isinstance(content, str)
                is_prev_str = isinstance(new_messages[-1]["content"], str)
                if is_current_str and is_prev_str:
                    new_messages[-1]["content"] += "\n\n" + content
                    continue
            prev_role = role
            new_messages.append(m)
        return new_messages

    @staticmethod
    def _format_chat(messages):
        for m in messages:
            content = m["content"]
            role = m["role"]
            if role == "user" and content is None:
                continue
            if role == "user" and isinstance(content, str) and m["user_name"]:
                m["content"] = "{}: {}".format(m["user_name"], content)
        return messages

    @staticmethod
    def _fix_broken_tool_calls(messages):
        clean_messages = []
        is_expecting_tool_answer = False
        for m in messages:
            if is_expecting_tool_answer and "tool_call_id" not in m:
                clean_messages = clean_messages[:-1]
            is_expecting_tool_answer = "tool_calls" in m
            clean_messages.append(m)
        if is_expecting_tool_answer:
            clean_messages = clean_messages[:-1]
        return clean_messages

    def _prepare_history(self, history, model: str, is_chat: bool = False):
        if is_chat:
            history = self._format_chat(history)
        assert history
        save_keys = ("content", "role", "tool_calls", "tool_call_id", "name")
        history = [{k: m[k] for k in save_keys if m.get(k) is not None or k == "content"} for m in history]
        history = [m for m in history if not self._is_image_content(m["content"]) or self.can_handle_images[model]]
        assert history
        history = [
            m for m in history if ("tool_calls" not in m and "tool_call_id" not in m) or self.can_handle_tools[model]
        ]
        assert history
        history = self._merge_messages(history)
        assert history
        tokens_count = self._count_tokens(history, model=model)
        while tokens_count > self.history_max_tokens and len(history) >= 3:
            history = history[2:]
            tokens_count = self._count_tokens(history, model=model)
        assert history
        return history

    def _get_user_name(self, user):
        return user.full_name if user.full_name else user.username

    def _crop_content(self, content):
        if isinstance(content, str):
            return content.replace("\n", " ")[:40]
        return IMAGE_PLACEHOLDER

    def _is_image_content(self, content):
        return isinstance(content, list) and content[-1]["type"] == "image_url"

    def _fix_image_roles(self, messages):
        for m in messages:
            if self._is_image_content(m["content"]):
                m["role"] = "user"
        return messages

    def _replace_images(self, messages):
        for m in messages:
            if self._is_image_content(m["content"]):
                m["content"] = IMAGE_PLACEHOLDER
        return messages

    def _truncate_text(self, text: str):
        if len(text) > self.chunk_size:
            text = text[: self.chunk_size] + "... truncated"
        return text


def main(
    bot_token: str,
    client_config_path: str,
    db_path: str,
    history_max_tokens: int = 6144,
    chunk_size: int = 3500,
    characters_path: str = None,
    tools_config_path: str = None,
) -> None:
    bot = LlmBot(
        bot_token=bot_token,
        client_config_path=client_config_path,
        db_path=db_path,
        history_max_tokens=history_max_tokens,
        chunk_size=chunk_size,
        characters_path=characters_path,
        tools_config_path=tools_config_path,
    )
    asyncio.run(bot.start_polling())


if __name__ == "__main__":
    fire.Fire(main)
