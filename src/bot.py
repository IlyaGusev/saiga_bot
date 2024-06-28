import asyncio
import io
import os
import json
import copy
import traceback
import base64
import textwrap
from functools import wraps
from email.utils import parseaddr
from typing import cast, List, Dict, Any, Optional, Union, Callable, Coroutine

import requests
import fire  # type: ignore
import tiktoken
from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode, ChatMemberStatus
from aiogram.filters import Command, CommandObject
from aiogram.types import Message, InlineKeyboardButton, CallbackQuery, BufferedInputFile, User
from aiogram.utils.keyboard import InlineKeyboardBuilder
from openai import AsyncOpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from transformers import AutoTokenizer  # type: ignore
from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type: ignore

from src.localization import Localization
from src.tools import Tool
from src.database import Database
from src.payments import YookassaHandler, YookassaStatus

os.environ["TOKENIZERS_PARALLELISM"] = "false"


DEFAULT_HISTORY_MAX_TOKENS = 6144
DEFAULT_MESSAGE_COUNT_LIMIT = {
    "standard": {"limit": 1000, "interval": 86400},
    "subscribed": {"limit": 1000, "interval": 86400},
}
TEMPERATURE_RANGE = (0.0, 0.5, 0.8, 1.0, 1.2)
TOP_P_RANGE = (0.8, 0.9, 0.95, 0.98, 1.0)
DALLE_DAILY_LIMIT = 5
CONTACT_USERNAME = "YallenGusev"
START_TEMPLATE = """
Привет! Я Сайга, бот с разными языковыми моделями.
Текущая модель: {model}
Осталось сообщений: {message_count}

Лимиты:
{sub_limits}

Доступные команды:
/help - вызов этого сообщения
/reset - очистить историю сообщений с ботом
/setmodel - выбрать модель
/getmodel - узнать текущую модель
/setcharacter - задать персонажа
/setsystem ... - задать системный промпт, нужно писать в этом же сообщении
/getsystem - узнать текущий системный промпт
/resetsystem - сбросить системный промпт к стандартному
/setshortname ... - задать имя бота, нужно писать в этом же сообщении
/getshortname - узнать имя ботя
/getcount - узнать текущий лимит сообщений
/getparams - узнать текущие параметры генерации
/settemperature - задать температуру
/subinfo - информация о текущей подписке
/subbuy - купить подписку
/setemail ... - задать e-mail, чтобы купить подписку, нужно писать в этом же сообщении
/tools - включить/выключить инструменты (плагины)

Инструменты, доступные для gpt-4o и claude-3-5-sonnet:
- Поиск в интернете
- Чтение страниц в интернете
- Текущая дата и время
- Генерация изображений с DALL-E
- Интерпретатор Python

Исходники: [saiga_bot](https://github.com/IlyaGusev/saiga_bot)
Модель по умолчанию: [saiga_llama3_8b](https://huggingface.co/IlyaGusev/saiga_llama3_8b)

По всем вопросам писать @{contact_username}
"""

IMAGE_PLACEHOLDER = "<image_placeholder>"

SUB_PRICE = 500
SUB_TITLE = 'Покупка подписки в боте "Сайга" на неделю для пользователя {user_id}'
SUB_DESCRIPTION = """*Покупка подписки в боте 'Сайга' на неделю*

Лимиты общения с моделями станут такими:
{sub_limits}

Подписка будет действовать ровно 7 дней.

Цена подписки: *{price} рублей*
"""

ChatMessage = Dict[str, Any]
ChatMessages = List[ChatMessage]


class Tokenizer:
    tokenizers: Dict[str, AutoTokenizer] = dict()

    @classmethod
    def get(cls, model_name: str) -> AutoTokenizer:
        if model_name not in cls.tokenizers:
            cls.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
        return cls.tokenizers[model_name]


def check_admin(func: Callable[..., Coroutine[Any, Any, Any]]) -> Callable[..., Coroutine[Any, Any, Any]]:
    @wraps(func)
    async def wrapped(self: Any, obj: Union[CallbackQuery, Message], *args: Any, **kwargs: Any) -> Any:
        if isinstance(obj, CallbackQuery):
            assert obj.message
            assert obj.from_user
            chat_id = obj.message.chat.id
            user_id = obj.from_user.id
            user = obj.from_user
        elif isinstance(obj, Message):
            assert obj.chat
            assert obj.from_user
            chat_id = obj.chat.id
            user_id = obj.from_user.id
            user = obj.from_user
        else:
            assert False
        assert user

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
        localization_config_path: str,
        chunk_size: Optional[int],
        characters_path: Optional[str],
        tools_config_path: Optional[str],
        yookassa_config_path: Optional[str],
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
        self.history_max_tokens = dict()
        self.limits = dict()
        for model_name, config in client_config.items():
            self.model_names[model_name] = config.pop("model_name")
            self.can_handle_images[model_name] = config.pop("can_handle_images", False)
            self.can_handle_tools[model_name] = config.pop("can_handle_tools", False)
            self.default_prompts[model_name] = config.pop("system_prompt", "")
            self.history_max_tokens[model_name] = config.pop("history_max_tokens", DEFAULT_HISTORY_MAX_TOKENS)
            if "params" in config:
                self.default_params[model_name] = config.pop("params")
            self.limits[model_name] = config.pop("message_count_limit", DEFAULT_MESSAGE_COUNT_LIMIT)
            assert "standard" in self.limits[model_name]
            assert "subscribed" in self.limits[model_name]
            self.clients[model_name] = AsyncOpenAI(**config)
        assert self.clients
        assert self.model_names
        assert self.default_prompts

        self.localization = Localization.load(localization_config_path, "ru")

        # Инструменты
        self.tools: Dict[str, Tool] = dict()
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

        # Бот
        self.bot = Bot(token=bot_token, default=DefaultBotProperties(parse_mode=None))
        self.bot_info: Optional[User] = None
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
        self.dp.message.register(self.set_email, Command("setemail"))
        self.dp.message.register(self.sub_info, Command("subinfo"))
        self.dp.message.register(self.sub_buy, Command("subbuy"))
        self.dp.message.register(self.toogle_tools, Command("tools"))
        self.dp.message.register(self.history, Command("history"))
        self.dp.message.register(self.generate)

        self.dp.callback_query.register(self.save_feedback, F.data.startswith("feedback:"))
        self.dp.callback_query.register(self.set_model_button_handler, F.data.startswith("setmodel:"))
        self.dp.callback_query.register(self.set_character_button_handler, F.data.startswith("setcharacter:"))
        self.dp.callback_query.register(self.set_temperature_button_handler, F.data.startswith("settemperature:"))
        self.dp.callback_query.register(self.set_top_p_button_handler, F.data.startswith("settopp:"))
        self.dp.callback_query.register(self.yookassa_sub_buy_proceed, F.data.startswith("buy:yookassa"))

        # Платежи
        self.scheduler: Optional[AsyncIOScheduler] = None
        self.yookassa: Optional[YookassaHandler] = None
        if yookassa_config_path and os.path.exists(yookassa_config_path):
            self.buy_kb.add(InlineKeyboardButton(text="Купить (из России)", callback_data="buy:yookassa"))
            with open(yookassa_config_path) as r:
                config = json.load(r)
                self.yookassa = YookassaHandler(**config)

    async def start_polling(self) -> None:
        self.scheduler = AsyncIOScheduler(timezone="Europe/Moscow")
        if self.yookassa is not None:
            self.scheduler.add_job(self.yookassa_check_payments, trigger="interval", seconds=30)
        self.scheduler.start()
        self.bot_info = await self.bot.get_me()
        await self.dp.start_polling(self.bot)

    async def start(self, message: Message) -> None:
        assert message.from_user
        user_id = message.from_user.id
        chat_id = message.chat.id
        self.db.create_conv_id(chat_id)
        model = self.db.get_current_model(chat_id)
        remaining_count = self._count_remaining_messages(user_id=user_id, model=model)
        sub_limits = self._get_limits()
        content = START_TEMPLATE.format(
            model=model, message_count=remaining_count, sub_limits=sub_limits, contact_username=CONTACT_USERNAME
        )
        await message.reply(content, parse_mode=ParseMode.MARKDOWN)

    #
    # История сообщений
    #

    async def reset(self, message: Message) -> None:
        chat_id = message.chat.id
        self.db.create_conv_id(chat_id)
        await message.reply(self.localization.RESET)

    async def history(self, message: Message) -> None:
        chat_id = message.chat.id
        assert message.from_user
        is_chat = chat_id != message.from_user.id
        conv_id = self.db.get_current_conv_id(chat_id)
        history = self.db.fetch_conversation(conv_id)
        model = self.db.get_current_model(chat_id)
        message_text = self.localization.NO_HISTORY
        if history:
            history = self._replace_images(history)
            history = self._prepare_history(history, model=model, is_chat=is_chat)
            tokens_count = self._count_tokens(history, model=model)
            plain_history = json.dumps(history, ensure_ascii=False)
            plain_history = self._truncate_text(plain_history)
            message_text = self.localization.HISTORY.format(tokens_count=tokens_count, history=plain_history)
        await message.reply(message_text)

    async def _save_chat_message(self, message: Message) -> None:
        chat_id = message.chat.id
        assert message.from_user
        user_id = message.from_user.id
        user_name = self._get_user_name(message.from_user)
        content = await self._build_content(message)
        if content is not None:
            conv_id = self.db.get_current_conv_id(chat_id)
            self.db.save_user_message(content, conv_id=conv_id, user_id=user_id, user_name=user_name)

    #
    # Выбор модели
    #

    @check_admin
    async def set_model(self, message: Message) -> None:
        await message.reply(self.localization.CHOOSE_MODEL, reply_markup=self.models_kb.as_markup())

    @check_admin
    async def set_model_button_handler(self, callback: CallbackQuery) -> None:
        assert callback.message
        assert callback.data
        chat_id = callback.message.chat.id
        model_name = callback.data.split(":")[1]
        assert model_name in self.clients
        self.db.set_current_model(chat_id, model_name)
        self.db.create_conv_id(chat_id)
        assert isinstance(callback.message, Message)
        await callback.message.edit_text(self.localization.NEW_MODEL.format(model_name=model_name))

    async def get_model(self, message: Message) -> None:
        chat_id = message.chat.id
        model = self.db.get_current_model(chat_id)
        await message.reply(model)

    #
    # Системный промпт
    #

    @check_admin
    async def set_system(self, message: Message, command: CommandObject) -> None:
        chat_id = message.chat.id
        text = command.args
        text = text if text else ""
        self.db.set_system_prompt(chat_id, text)
        self.db.create_conv_id(chat_id)
        text = self._truncate_text(text)
        await message.reply(self.localization.NEW_SYSTEM_PROMPT.format(system_prompt=text))

    async def get_system(self, message: Message) -> None:
        chat_id = message.chat.id
        prompt = self.db.get_system_prompt(chat_id, self.default_prompts)
        if not prompt.strip():
            prompt = self.localization.EMPTY_SYSTEM_PROMPT
        await message.reply(prompt)

    async def reset_system(self, message: Message) -> None:
        chat_id = message.chat.id
        model = self.db.get_current_model(chat_id)
        self.db.set_system_prompt(chat_id, self.default_prompts.get(model, ""))
        self.db.create_conv_id(chat_id)
        await message.reply(self.localization.RESET_SYSTEM_PROMPT)

    #
    # Именование модели
    #

    @check_admin
    async def set_short_name(self, message: Message, command: CommandObject) -> None:
        chat_id = message.chat.id
        text = command.args
        text = text if text else ""
        text = text.strip()
        message_text = self.localization.EMPTY_SHORT_NAME
        if text:
            self.db.set_short_name(chat_id, text)
            self.db.create_conv_id(chat_id)
            message_text = self.localization.NEW_SHORT_NAME.format(name=text)
        await message.reply(message_text)

    async def get_short_name(self, message: Message) -> None:
        chat_id = message.chat.id
        name = self.db.get_short_name(chat_id)
        text = self.localization.GET_SHORT_NAME.format(name=name)
        await message.reply(text)

    #
    # Персонажи
    #

    @check_admin
    async def set_character(self, message: Message) -> None:
        await message.reply(self.localization.CHOOSE_CHARACTER, reply_markup=self.characters_kb.as_markup())

    @check_admin
    async def set_character_button_handler(self, callback: CallbackQuery) -> None:
        assert callback.message
        assert callback.data
        chat_id = callback.message.chat.id
        char_name = callback.data.split(":")[1]
        assert char_name in self.characters
        character = self.characters[char_name]
        system_prompt = character["system_prompt"]
        short_name = character["short_name"]
        self.db.set_system_prompt(chat_id, system_prompt)
        self.db.set_short_name(chat_id, short_name)
        self.db.create_conv_id(chat_id)
        assert isinstance(callback.message, Message)
        await callback.message.edit_text(
            self.localization.NEW_CHARACTER.format(system_prompt=system_prompt, name=short_name)
        )

    #
    # Лимиты
    #

    def _count_remaining_messages(self, user_id: int, model: str) -> int:
        is_subscribed = self.db.is_subscribed_user(user_id)
        mode = "standard" if not is_subscribed else "subscribed"
        limit = int(self.limits[model][mode]["limit"])
        interval = self.limits[model][mode]["interval"]
        count = int(self.db.count_user_messages(user_id, model, interval))
        remaining_count = limit - count
        return max(0, remaining_count)

    async def get_count(self, message: Message) -> None:
        assert message.from_user
        user_id = message.from_user.id
        chat_id = message.chat.id
        model = self.db.get_current_model(chat_id)
        remaining_count = self._count_remaining_messages(user_id=user_id, model=model)
        text = self.localization.REMAINING_MESSAGES.format(model=model, remaining_count=remaining_count)
        await message.reply(text)

    async def sub_info(self, message: Message) -> None:
        assert message.from_user
        user_id = message.from_user.id
        remaining_seconds = self.db.get_subscription_info(user_id)
        text = self.localization.INACTIVE_SUB
        if remaining_seconds > 0:
            text = self.localization.ACTIVE_SUB.format(remaining_seconds=remaining_seconds)
        await message.reply(text)

    def _get_limits(self) -> str:
        template = "- *{model}*: {count} сообщений каждые {hours} часа"
        sub_limits = [
            template.format(
                model=model, count=limit["subscribed"]["limit"], hours=limit["subscribed"]["interval"] // 3600
            )
            for model, limit in self.limits.items()
        ]
        return "\n".join(sub_limits)

    async def sub_buy(self, message: Message) -> None:
        assert message.from_user
        user_id = message.from_user.id
        email = self.db.get_email(user_id)
        if not email:
            await message.reply(self.localization.SET_EMAIL)
            return

        chat_id = message.chat.id
        is_chat = chat_id != user_id
        if is_chat:
            await message.reply("Подписку можно купить только в переписке с самим ботом!")
            return

        remaining_seconds = self.db.get_subscription_info(user_id)
        if remaining_seconds > 0:
            await message.reply(f"У вас уже есть подписка! Она закончится через {remaining_seconds//3600}ч")
            return

        sub_limits = self._get_limits()
        description = SUB_DESCRIPTION.format(sub_limits=sub_limits, price=SUB_PRICE)
        await message.reply(description, parse_mode=ParseMode.MARKDOWN, reply_markup=self.buy_kb.as_markup())

    async def yookassa_sub_buy_proceed(self, callback: CallbackQuery) -> None:
        assert self.yookassa
        assert callback.from_user
        assert callback.message
        assert isinstance(callback.message, Message)
        user_id = callback.from_user.id
        email = self.db.get_email(user_id)
        if not email:
            await callback.message.reply(self.localization.SET_EMAIL)
            return

        chat_id = callback.message.chat.id
        is_chat = chat_id != user_id
        if is_chat:
            await callback.message.reply("Подписку можно купить только в переписке с самим ботом!")
            return

        remaining_seconds = self.db.get_subscription_info(user_id)
        if remaining_seconds > 0:
            await callback.message.reply(f"У вас уже есть подписка! Она закончится через {remaining_seconds//3600}ч")
            return

        timestamp = self.db.get_current_ts()
        title = SUB_TITLE.format(user_id=user_id)
        assert self.bot_info
        assert self.bot_info.username
        payment_data = self.yookassa.create_payment(SUB_PRICE, title, email=email, bot_username=self.bot_info.username)
        payment_id = payment_data["id"]
        try:
            url = payment_data["confirmation"]["confirmation_url"]
            status = payment_data["status"]
            self.db.save_payment(
                payment_id=payment_id, user_id=user_id, chat_id=chat_id, url=url, status=status, timestamp=timestamp
            )
            await callback.message.reply(f"Ссылка для оплаты: {url}")
        except Exception:
            self.yookassa.cancel_payment(payment_id)

    async def yookassa_check_payments(self) -> None:
        assert self.yookassa

        payments = self.db.get_waiting_payments()

        for payment in payments:
            status = self.yookassa.check_payment(payment.payment_id)
            self.db.set_payment_status(
                payment_id=payment.payment_id, status=status, internal_status=payment.internal_status
            )
            if status == YookassaStatus.SUCCEEDED:
                self.db.subscribe_user(payment.user_id, 7 * 86400)
                text = "Платёж получен, подписка выдана! Узнать статус: /subinfo"
                await self.bot.send_message(chat_id=payment.chat_id, text=text)
                self.db.set_payment_status(payment.payment_id, status=status.value, internal_status="completed")
            elif status == YookassaStatus.CANCELED:
                await self.bot.send_message(chat_id=payment.chat_id, text="Платёж отменён!")
                self.db.set_payment_status(payment.payment_id, status=status.value, internal_status="completed")

    async def set_email(self, message: Message) -> None:
        assert message.text
        assert message.from_user
        email = message.text.replace("/setemail", "").strip()
        is_valid = "@" in parseaddr(email)[1]
        if not is_valid:
            await message.reply("Некорректный e-mail!")
            return
        self.db.set_email(message.from_user.id, email)
        await message.reply(f"Спасибо! Адрес задан: {email}")

    #
    # Параметры генерации
    #

    @check_admin
    async def set_temperature(self, message: Message) -> None:
        await message.reply("Выберите температуру:", reply_markup=self.temperature_kb.as_markup())

    @check_admin
    async def set_temperature_button_handler(self, callback: CallbackQuery) -> None:
        assert callback.message
        assert callback.data
        chat_id = callback.message.chat.id
        temperature = float(callback.data.split(":")[1])
        self.db.set_parameters(chat_id, self.default_params, temperature=temperature)
        assert isinstance(callback.message, Message)
        await callback.message.edit_text(f"Новая температура задана:\n\n{temperature}")

    @check_admin
    async def set_top_p(self, message: Message) -> None:
        await message.reply("Выберите top-p:", reply_markup=self.top_p_kb.as_markup())

    @check_admin
    async def set_top_p_button_handler(self, callback: CallbackQuery) -> None:
        assert callback.message
        assert callback.data
        chat_id = callback.message.chat.id
        top_p = float(callback.data.split(":")[1])
        self.db.set_parameters(chat_id, self.default_params, top_p=top_p)
        assert isinstance(callback.message, Message)
        await callback.message.edit_text(f"Новое top-p задано:\n\n{top_p}")

    async def get_params(self, message: Message) -> None:
        chat_id = message.chat.id
        params = self.db.get_parameters(chat_id, self.default_params)
        await message.reply(f"Текущие параметры генерации: {json.dumps(params)}")

    #
    # Инструменты
    #

    def _get_tools(self, chat_id: int) -> Optional[List[Dict[str, Any]]]:
        model = self.db.get_current_model(chat_id)
        if self.can_handle_tools[model] and self.tools and self.db.are_tools_enabled(chat_id):
            return [t.get_specification() for t in self.tools.values()]
        return None

    async def toogle_tools(self, message: Message) -> None:
        chat_id = message.chat.id
        model = self.db.get_current_model(chat_id)
        if not self.can_handle_tools[model]:
            await message.reply(f"Для модели {model} инструменты недоступны.")
            return
        current_value = self.db.are_tools_enabled(chat_id)
        self.db.set_enable_tools(chat_id, not current_value)
        if not current_value:
            await message.reply("Инструменты включены!")
        else:
            await message.reply("Инструменты выключены! Чтобы включить их назад, снова наберите /tools")

    async def _check_tools(self, messages: ChatMessages, model: str) -> Any:
        messages = copy.deepcopy(messages)
        messages = self._replace_images(messages)
        messages = self._fix_broken_tool_calls(messages)
        tools = [t.get_specification() for t in self.tools.values()]
        casted_messages = [cast(ChatCompletionMessageParam, message) for message in messages]
        casted_tools = [cast(ChatCompletionToolParam, tool) for tool in tools]
        chat_completion = await self.clients[model].chat.completions.create(
            model=self.model_names[model],
            messages=casted_messages,
            tools=casted_tools,
            tool_choice="auto",
            max_tokens=2048,
        )
        response_message = chat_completion.choices[0].message
        return response_message

    async def _call_dalle(self, conv_id: str, user_id: int, chat_id: int, placeholder: Message, **kwargs: Any) -> None:
        dalle_count = self.db.count_generated_images(user_id, 86400)
        is_dalle_remaining = DALLE_DAILY_LIMIT - dalle_count > 0
        if not is_dalle_remaining:
            await placeholder.edit_text("Лимит по генерации картинок исчерпан, восстановится через 24 часа")
            return

        await placeholder.edit_text(f"Генерирую картинку по промпту: {kwargs['prompt_russian']}")
        function_response = await self.tools["dalle"](**kwargs)
        if not isinstance(function_response, list) or "image_url" not in function_response[1]:
            text = f"Ошибка при вызове DALL-E: {function_response}"
            new_message = await self.bot.send_message(
                chat_id=chat_id, reply_to_message_id=placeholder.message_id, text=text
            )
            self.db.save_assistant_message(
                content=text,
                conv_id=conv_id,
                message_id=new_message.message_id,
                model="dalle",
                reply_user_id=user_id,
            )
            return
        image_url: Any = function_response[1]["image_url"]
        base64_image = image_url["url"].replace("data:image/jpeg;base64,", "")
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

    async def _call_tools(
        self, history: ChatMessages, model: str, conv_id: str, user_id: int, chat_id: int, placeholder: Message
    ) -> Optional[ChatMessages]:
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

            function_response: Union[str, List[Dict[str, Any]], None] = None
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

            assert function_response
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

    async def generate(self, message: Message) -> None:
        assert message.from_user
        user_id = message.from_user.id
        user_name = self._get_user_name(message.from_user)
        chat_id = user_id
        is_chat = False
        if message.chat.type in ("group", "supergroup"):
            chat_id = message.chat.id
            is_chat = True
            assert self.bot_info
            is_reply = (
                message.reply_to_message
                and message.reply_to_message.from_user
                and message.reply_to_message.from_user.id == self.bot_info.id
            )
            bot_short_name = self.db.get_short_name(chat_id)
            assert self.bot_info.username
            bot_names = ["@" + self.bot_info.username, bot_short_name]
            is_explicit = message.text and any(bot_name in message.text for bot_name in bot_names)
            if not is_reply and not is_explicit:
                await self._save_chat_message(message)
                return

        model = self.db.get_current_model(chat_id)
        if model not in self.clients:
            await message.reply("Выбранная модель больше не поддерживается, переключите на другую с помощью /setmodel")
            return

        remaining_count = self._count_remaining_messages(user_id=user_id, model=model)
        print(user_id, model, remaining_count)
        if remaining_count <= 0:
            await message.reply(
                f"Превышен дневной лимит запросов по {model}, подождите или переключите модель с /setmodel"
            )
            return

        params = self.db.get_parameters(chat_id, self.default_params)
        assert params
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
            tools = self._get_tools(chat_id)
            if tools:
                response = await self._call_tools(
                    history=history,
                    model=model,
                    user_id=user_id,
                    conv_id=conv_id,
                    chat_id=chat_id,
                    placeholder=placeholder,
                )
                if response is None:
                    return
                history = response

            history = self._fix_image_roles(history)
            history = self._fix_broken_tool_calls(history)
            if tools and "gpt" not in model:
                params["tools"] = tools
            answer = await self._query_api(model=model, messages=history, system_prompt=system_prompt, **params)

            chunk_size = self.chunk_size
            if chunk_size is not None:
                answer_parts = [answer[i : i + chunk_size] for i in range(0, len(answer), chunk_size)]
            else:
                answer_parts = [answer]

            new_message = await placeholder.edit_text(answer_parts[0])
            assert isinstance(new_message, Message)
            for part in answer_parts[1:]:
                new_message = await message.reply(part)
                assert isinstance(new_message, Message)

            markup = self.likes_kb.as_markup()
            new_message = await new_message.edit_text(
                answer_parts[-1],
                reply_markup=markup,
            )
            assert isinstance(new_message, Message)
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
            text = "Что-то пошло не так, ответ от Сайги не получен или не смог отобразиться."
            text += f" Попробуйте сделать /reset и пришлите @{CONTACT_USERNAME} вот это число: {chat_id}"
            await placeholder.edit_text(text)

    async def _query_api(self, model: str, messages: ChatMessages, system_prompt: str, **kwargs: Any) -> str:
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
        casted_messages = [cast(ChatCompletionMessageParam, message) for message in messages]
        chat_completion = await self.clients[model].chat.completions.create(
            model=self.model_names[model], messages=casted_messages, **kwargs
        )
        assert chat_completion.choices, str(chat_completion)
        assert chat_completion.choices[0].message.content, str(chat_completion)
        assert isinstance(chat_completion.choices[0].message.content, str), str(chat_completion)
        answer: str = chat_completion.choices[0].message.content
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

    async def _build_content(self, message: Message) -> Union[None, str, List[Dict[str, Any]]]:
        content_type = message.content_type
        if content_type == "text":
            assert message.text
            text = message.text
            chat_id = message.chat.id
            bot_short_name = self.db.get_short_name(chat_id)
            assert self.bot_info
            assert self.bot_info.username
            text = text.replace("@" + self.bot_info.username, bot_short_name).strip()
            return text

        photo = None
        photo_ext = (".jpg", "jpeg", ".png", ".webp", ".gif")
        if content_type == "photo":
            assert message.photo
            file_info = await self.bot.get_file(message.photo[-1].file_id)
            photo = file_info.file_path
        elif content_type == "document":
            document = message.document
            if document:
                file_info = await self.bot.get_file(document.file_id)
                if file_info and file_info.file_path:
                    file_path = file_info.file_path
                    if "." + file_path.split(".")[-1].lower() in photo_ext:
                        photo = file_path

        if photo:
            file_stream = await self.bot.download_file(photo)
            assert file_stream
            file_stream.seek(0)
            base64_image = base64.b64encode(file_stream.read()).decode("utf-8")
            assert base64_image
            content: List[Dict[str, Any]] = []
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

    async def save_feedback(self, callback: CallbackQuery) -> None:
        assert callback.from_user
        assert callback.message
        assert callback.data
        user_id = callback.from_user.id
        message_id = callback.message.message_id
        feedback = callback.data.split(":")[1]
        self.db.save_feedback(feedback, user_id=user_id, message_id=message_id)
        await self.bot.edit_message_reply_markup(
            chat_id=callback.message.chat.id, message_id=message_id, reply_markup=None
        )

    def _count_tokens(self, messages: ChatMessages, model: str) -> int:
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

    async def _is_admin(self, user_id: int, chat_id: int) -> bool:
        chat_member = await self.bot.get_chat_member(chat_id, user_id)
        return chat_member.status in [
            ChatMemberStatus.ADMINISTRATOR,
            ChatMemberStatus.CREATOR,
        ]

    @staticmethod
    def _merge_messages(messages: ChatMessages) -> ChatMessages:
        new_messages: ChatMessages = []
        prev_role = None
        for m in messages:
            content = m["content"]
            role = m["role"]
            if role == prev_role and role != "tool":
                is_current_str = isinstance(content, str)
                is_prev_str = isinstance(new_messages[-1]["content"], str)
                if is_current_str and is_prev_str:
                    new_messages[-1]["content"] += "\n\n" + content
                    continue
            prev_role = role
            new_messages.append(m)
        return new_messages

    @staticmethod
    def _format_chat(messages: ChatMessages) -> ChatMessages:
        for m in messages:
            content = m["content"]
            role = m["role"]
            if role == "user" and content is None:
                continue
            if role == "user" and isinstance(content, str) and m["user_name"]:
                m["content"] = "Из чата пишет {}: {}".format(m["user_name"], content)
        return messages

    @staticmethod
    def _fix_broken_tool_calls(messages: ChatMessages) -> ChatMessages:
        clean_messages: ChatMessages = []
        is_expecting_tool_answer = False
        for m in messages:
            if is_expecting_tool_answer and "tool_call_id" not in m:
                clean_messages = clean_messages[:-1]
            is_expecting_tool_answer = "tool_calls" in m
            clean_messages.append(m)
        if is_expecting_tool_answer:
            clean_messages = clean_messages[:-1]
        return clean_messages

    def _prepare_history(self, history: ChatMessages, model: str, is_chat: bool = False) -> ChatMessages:
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
        while tokens_count > self.history_max_tokens[model] and len(history) >= 3:
            history = history[2:]
            tokens_count = self._count_tokens(history, model=model)
        assert history
        return history

    def _get_user_name(self, user: User) -> str:
        return str(user.full_name) if user.full_name else str(user.username)

    def _crop_content(self, content: str) -> str:
        if isinstance(content, str):
            return content.replace("\n", " ")[:40]
        return IMAGE_PLACEHOLDER

    def _is_image_content(self, content: Any) -> bool:
        return isinstance(content, list) and content[-1]["type"] == "image_url"

    def _fix_image_roles(self, messages: ChatMessages) -> ChatMessages:
        for m in messages:
            if self._is_image_content(m["content"]):
                m["role"] = "user"
        return messages

    def _replace_images(self, messages: ChatMessages) -> ChatMessages:
        for m in messages:
            if self._is_image_content(m["content"]):
                m["content"] = IMAGE_PLACEHOLDER
        return messages

    def _truncate_text(self, text: str) -> str:
        if self.chunk_size and len(text) > self.chunk_size:
            text = text[: self.chunk_size] + "... truncated"
        return text


def main(
    bot_token: str,
    client_config_path: str,
    db_path: str,
    localization_config_path: str,
    chunk_size: Optional[int] = 3500,
    characters_path: Optional[str] = None,
    tools_config_path: Optional[str] = None,
    yookassa_config_path: Optional[str] = None,
) -> None:
    bot = LlmBot(
        bot_token=bot_token,
        client_config_path=client_config_path,
        db_path=db_path,
        chunk_size=chunk_size,
        characters_path=characters_path,
        tools_config_path=tools_config_path,
        yookassa_config_path=yookassa_config_path,
        localization_config_path=localization_config_path,
    )
    asyncio.run(bot.start_polling())


if __name__ == "__main__":
    fire.Fire(main)
