import asyncio
import os
import json
import copy
import traceback
import base64
import io
import re
from email.utils import parseaddr
from enum import Enum
from typing import cast, List, Dict, Any, Optional, Union, Callable, Coroutine, Tuple, BinaryIO
from dataclasses import dataclass, field

import fire  # type: ignore
import tiktoken
from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode, ChatMemberStatus
from aiogram.filters import Command, CommandObject
from aiogram.types import (
    Message,
    InlineKeyboardButton,
    CallbackQuery,
    BufferedInputFile,
    User,
    ContentType,
    PreCheckoutQuery,
    LabeledPrice,
    SuccessfulPayment,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.exceptions import TelegramBadRequest
from openai import AsyncOpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type: ignore

from src.provider import LLMProvider
from src.llm_filter import LLMFilter
from src.decorators import check_admin, check_creator
from src.localization import Localization
from src.tools import Tool
from src.database import Database
from src.payments import YookassaHandler, YookassaStatus
from src.tokenizers import Tokenizers
from src.document_loader import DocumentLoader


DALLE_DAILY_LIMIT = 5
IMAGE_PLACEHOLDER = "<image_placeholder>"

ChatMessage = Dict[str, Any]
ChatMessages = List[ChatMessage]

@dataclass
class SubConfig:
    price: int = 500
    currency: str = "RUB"
    duration: int = 7 * 86400


class SubKey(str, Enum):
    RUB_WEEK = "rub_week"
    RUB_MONTH = "rub_month"
    XTR_WEEK = "xtr_week"
    XTR_MONTH = "xtr_month"


@dataclass
class BotConfig:
    token: str
    admin_user_name: str
    admin_user_id: int
    temperature_range: List[float]
    top_p_range: List[float]
    freq_penalty_range: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.1, 0.2, 0.5, 1.0])
    timezone: str = "Europe/Moscow"
    output_chunk_size: int = 3500
    sub_configs: Dict[SubKey, SubConfig] = field(default_factory=lambda: {
        SubKey.RUB_WEEK: SubConfig(700, "RUB", 7 * 86400),
        SubKey.RUB_MONTH: SubConfig(2800, "RUB", 31 * 86400),
        SubKey.XTR_WEEK:  SubConfig(500, "XTR", 7 * 86400),
        SubKey.XTR_MONTH:  SubConfig(1500, "XTR", 31 * 86400),
    })


def _crop_content(content: str) -> str:
    if isinstance(content, str):
        return content.replace("\n", " ")[:40]
    return IMAGE_PLACEHOLDER


def _split_message(text: str, output_chunk_size: int) -> List[str]:
    if len(text) <= output_chunk_size:
        return [text]

    chunks: List[str] = []
    paragraphs = text.split("\n\n")
    for paragraph in paragraphs:
        if chunks and len(chunks[-1]) + len(paragraph) + 2 <= output_chunk_size:
            chunks[-1] += '\n\n' + paragraph
        else:
            chunks.append(paragraph)

    final_chunks: List[str] = []
    for chunk in chunks:
        if len(chunk) <= output_chunk_size:
            final_chunks.append(chunk)
            continue
        parts = [chunk[i : i + output_chunk_size] for i in range(0, len(chunk), output_chunk_size)]
        final_chunks.extend(parts)

    return final_chunks


async def _reply(message: Message, text: str, **kwargs: Any) -> Union[Message, bool]:
    try:
        return await message.reply(text, parse_mode=ParseMode.MARKDOWN, **kwargs)
    except Exception:
        try:
            return await message.reply(text, parse_mode=ParseMode.HTML, **kwargs)
        except Exception:
            return await message.reply(text, parse_mode=None, **kwargs)


async def _edit_text(message: Message, text: str, **kwargs: Any) -> Union[Message, bool]:
    try:
        return await message.edit_text(text, parse_mode=ParseMode.MARKDOWN, **kwargs)
    except Exception:
        try:
            return await message.edit_text(text, parse_mode=ParseMode.HTML, **kwargs)
        except Exception:
            return await message.edit_text(text, parse_mode=None, **kwargs)


class LlmBot:
    def __init__(
        self,
        providers_config_path: str,
        db_path: str,
        bot_config_path: str,
        localization_config_path: str,
        characters_path: Optional[str],
        tools_config_path: Optional[str],
        yookassa_config_path: Optional[str],
    ):
        assert os.path.exists(bot_config_path)
        with open(bot_config_path) as r:
            self.config = BotConfig(**json.load(r))

        self.providers: Dict[str, LLMProvider] = dict()
        with open(providers_config_path) as r:
            providers_config = json.load(r)
            for provider_name, config in providers_config.items():
                self.providers[provider_name] = LLMProvider(provider_name=provider_name, **config)

        self.llm_filter = None
        if "gpt-4o-mini" in self.providers:
            self.llm_filter = LLMFilter(self.providers["gpt-4o-mini"])

        self.localization = Localization.load(localization_config_path, "ru")

        self.tools: Dict[str, Tool] = dict()
        if tools_config_path:
            assert os.path.exists(tools_config_path)
            with open(tools_config_path) as r:
                tools_config = json.load(r)
                for tool_name, tool_config in tools_config.items():
                    self.tools[tool_name] = Tool.by_name(tool_name)(**tool_config)

        self.characters = dict()
        if characters_path and os.path.exists(characters_path):
            with open(characters_path) as r:
                self.characters = json.load(r)

        self.db = Database(db_path)
        self.dp = self.build_dispatcher()
        self.document_loader = DocumentLoader()
        self.bot = Bot(token=self.config.token, default=DefaultBotProperties(parse_mode=None))
        self.bot_info: Optional[User] = None
        self.scheduler: Optional[AsyncIOScheduler] = None

        self.yookassa: Optional[YookassaHandler] = None
        if yookassa_config_path and os.path.exists(yookassa_config_path):
            with open(yookassa_config_path) as r:
                config = json.load(r)
                self.yookassa = YookassaHandler(**config)
        self.build_menus(add_yookassa=self.yookassa is not None)

    def build_menus(self, add_yookassa: bool = False) -> None:
        self.models_kb = InlineKeyboardBuilder()
        for model_id in self.providers.keys():
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
        for value in self.config.temperature_range:
            self.temperature_kb.add(InlineKeyboardButton(text=str(value), callback_data=f"settemperature:{value}"))

        self.top_p_kb = InlineKeyboardBuilder()
        for value in self.config.top_p_range:
            self.top_p_kb.add(InlineKeyboardButton(text=str(value), callback_data=f"settopp:{value}"))

        self.freq_penalty_kb = InlineKeyboardBuilder()
        for value in self.config.freq_penalty_range:
            self.freq_penalty_kb.add(InlineKeyboardButton(text=str(value), callback_data=f"setfreqpenalty:{value}"))

        self.buy_kb = InlineKeyboardBuilder()
        self.buy_kb.add(InlineKeyboardButton(text=self.localization.BUY_WEEK_WITH_STARS, callback_data="buy:stars:xtr_week"))
        self.buy_kb.add(InlineKeyboardButton(text=self.localization.BUY_MONTH_WITH_STARS, callback_data="buy:stars:xtr_month"))
        if add_yookassa:
            self.buy_kb.add(InlineKeyboardButton(text=self.localization.BUY_WEEK_WITH_RUB, callback_data="buy:yookassa:rub_week"))
            self.buy_kb.add(InlineKeyboardButton(text=self.localization.BUY_MONTH_WITH_RUB, callback_data="buy:yookassa:rub_month"))
        self.buy_kb.adjust(2)

    def build_dispatcher(self) -> Dispatcher:
        dp = Dispatcher()
        commands: List[Tuple[str, Callable[..., Any]]] = [
            ("start", self.start),
            ("help", self.start),
            ("reset", self.reset),
            ("setsystem", self.set_system),
            ("getsystem", self.get_system),
            ("resetsystem", self.reset_system),
            ("setmodel", self.set_model),
            ("getmodel", self.get_model),
            ("setshortname", self.set_short_name),
            ("getshortname", self.get_short_name),
            ("setcharacter", self.set_character),
            ("getcount", self.get_count),
            ("getparams", self.get_params),
            ("settemperature", self.set_temperature),
            ("settopp", self.set_top_p),
            ("setfrequencypenalty", self.set_frequency_penalty),
            ("setemail", self.set_email),
            ("subinfo", self.sub_info),
            ("subbuy", self.sub_buy),
            ("tools", self.toogle_tools),
            ("history", self.history),
            ("debug", self.debug),
            ("privacy", self.privacy),
            ("paysupport", self.pay_support),
        ]
        for command, func in commands:
            dp.message.register(func, Command(command))
        dp.message.register(self.wrong_command, Command(re.compile(r"\S+")))
        dp.message.register(
            self.successful_payment_handler, lambda m: m.content_type == ContentType.SUCCESSFUL_PAYMENT
        )
        callbacks: List[Tuple[str, Callable[..., Any]]] = [
            ("feedback:", self.save_feedback_handler),
            ("setmodel:", self.set_model_button_handler),
            ("setcharacter:", self.set_character_button_handler),
            ("settemperature:", self.set_temperature_button_handler),
            ("settopp:", self.set_top_p_button_handler),
            ("setfreqpenalty:", self.set_frequency_penalty_button_handler),
            ("buy:yookassa", self.yookassa_sub_buy_proceed),
            ("buy:stars", self.stars_sub_buy_proceed),
        ]
        for start, func in callbacks:
            dp.callback_query.register(func, F.data.startswith(start))
        dp.pre_checkout_query.register(self.pre_checkout_handler)
        dp.message.register(self.generate)
        return dp

    async def start_polling(self) -> None:
        self.scheduler = AsyncIOScheduler(timezone=self.config.timezone)
        if self.yookassa is not None:
            self.scheduler.add_job(self.yookassa_check_payments, trigger="interval", seconds=30)
        self.scheduler.start()
        self.bot_info = await self.bot.get_me()
        await self.dp.start_polling(self.bot)

    async def start(self, message: Message) -> None:
        assert message.from_user
        chat_id = message.chat.id
        model = self.db.get_current_model(chat_id)
        if model not in self.providers:
            await message.reply(self.localization.MODEL_NOT_SUPPORTED)
            return

        self.db.create_conv_id(chat_id)
        user_id = message.from_user.id
        remaining_count = self._count_remaining_messages(user_id=user_id, model=model)
        mode = "standard" if self.db.get_subscription_info(user_id) <= 0 else "subscribed"
        limits = {name: provider.limits for name, provider in self.providers.items()}
        sub_limits = self.localization.LIMITS.render(limits=limits, mode=mode).strip()
        content = self.localization.HELP.render(
            model=model,
            message_count=remaining_count,
            sub_limits=sub_limits,
            admin_username=self.config.admin_user_name,
        )
        await message.reply(content, parse_mode=ParseMode.MARKDOWN)

    async def wrong_command(self, message: Message) -> None:
        chat_id = message.chat.id
        assert message.from_user
        is_chat = chat_id != message.from_user.id
        if not is_chat:
            await message.reply(self.localization.WRONG_COMMAND)

    async def pay_support(self, message: Message) -> None:
        await message.reply(self.localization.PAY_SUPPORT)

    async def privacy(self, message: Message) -> None:
        await message.reply(self.localization.PRIVACY.render(admin_username=self.config.admin_user_name))

    #
    # History management
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
        provider = self.providers[model]
        message_text = self.localization.NO_HISTORY
        if history:
            history = self._replace_images(history)
            history = self._prepare_history(history, provider, is_chat)
            tokens_count = self._count_tokens(history, provider)
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

    def _prepare_history(self, history: ChatMessages, provider: LLMProvider, is_chat: bool = False) -> ChatMessages:
        if is_chat:
            history = self._format_chat(history)
        assert history
        save_keys = ("content", "role", "tool_calls", "tool_call_id", "name")
        history = [{k: m[k] for k in save_keys if m.get(k) is not None or k == "content"} for m in history]
        history = [m for m in history if not self._is_image_content(m["content"]) or provider.can_handle_images]
        assert history
        history = [
            m for m in history if ("tool_calls" not in m and "tool_call_id" not in m) or provider.can_handle_tools
        ]
        assert history
        history = self._merge_messages(history)
        assert history
        tokens_count = self._count_tokens(history, provider)
        while tokens_count > provider.history_max_tokens and len(history) >= 3:
            history = history[2:]
            tokens_count = self._count_tokens(history, provider)
        last_message = history[-1]["content"]
        if tokens_count > provider.history_max_tokens and isinstance(last_message, str):
            estimated_char_count = provider.history_max_tokens * 2
            if len(last_message) > estimated_char_count:
                history[-1]["content"] = "<truncated text>\n\n" + last_message[-estimated_char_count:]
        assert history
        return history

    @staticmethod
    def _merge_messages(messages: ChatMessages) -> ChatMessages:
        new_messages: ChatMessages = []
        prev_role = None
        for m in messages:
            content = m["content"]
            role = m["role"]
            if role == prev_role and role != "tool":
                is_current_str = isinstance(content, str)
                is_current_list = isinstance(content, list)
                prev_content = new_messages[-1]["content"]
                is_prev_str = isinstance(prev_content, str)
                is_prev_list = isinstance(prev_content, list)
                if is_current_str and is_prev_str:
                    new_messages[-1]["content"] += "\n\n" + content
                    continue
                elif is_current_str and is_prev_list:
                    prev_content.append({"type": "text", "text": content})
                    continue
                elif is_prev_str and is_current_list:
                    content.insert(0, {"type": "text", "text": prev_content})
                    new_messages[-1]["content"] = content
                    continue
                elif is_current_list and is_prev_list:
                    prev_content.extend(content)
                    continue
            prev_role = role
            new_messages.append(m)
        return new_messages

    #
    # Model selection
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
        assert model_name in self.providers
        self.db.set_current_model(chat_id, model_name)
        self.db.create_conv_id(chat_id)
        assert isinstance(callback.message, Message)
        await callback.message.edit_text(self.localization.NEW_MODEL.format(model_name=model_name))

    async def get_model(self, message: Message) -> None:
        chat_id = message.chat.id
        model = self.db.get_current_model(chat_id)
        await message.reply(model)

    #
    # System prompt
    #

    @check_admin
    async def set_system(self, message: Message, command: CommandObject) -> None:
        chat_id = message.chat.id
        text = command.args
        text = text if text else ""
        self.db.set_system_prompt(chat_id, text)
        self.db.create_conv_id(chat_id)
        text = self._truncate_text(text)
        if text.strip():
            await message.reply(self.localization.NEW_SYSTEM_PROMPT.format(system_prompt=text))
        else:
            await message.reply(self.localization.EMPTY_SYSTEM_PROMPT)

    async def get_system(self, message: Message) -> None:
        chat_id = message.chat.id
        model = self.db.get_current_model(chat_id)
        prompt = self.db.get_system_prompt(chat_id)
        if prompt is None:
            prompt = self.providers[model].system_prompt
        if not prompt.strip():
            prompt = self.localization.EMPTY_SYSTEM_PROMPT
        await message.reply(prompt)

    @check_admin
    async def reset_system(self, message: Message) -> None:
        chat_id = message.chat.id
        model = self.db.get_current_model(chat_id)
        self.db.set_system_prompt(chat_id, self.providers[model].system_prompt)
        self.db.create_conv_id(chat_id)
        await message.reply(self.localization.RESET_SYSTEM_PROMPT)

    #
    # Model names
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
    # Characters
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
    # Limits
    #

    def _count_remaining_messages(self, user_id: int, model: str) -> int:
        is_subscribed = self.db.is_subscribed_user(user_id)
        mode = "standard" if not is_subscribed else "subscribed"
        provider = self.providers[model]
        limit = int(provider.limits[mode]["limit"])
        interval = provider.limits[mode]["interval"]
        count = int(self.db.count_user_messages(user_id, model, interval))
        remaining_count = limit - count
        return max(0, remaining_count)

    async def get_count(self, message: Message) -> None:
        assert message.from_user
        user_id = message.from_user.id
        chat_id = message.chat.id
        model = self.db.get_current_model(chat_id)
        if model not in self.providers:
            await message.reply(self.localization.MODEL_NOT_SUPPORTED)
            return
        remaining_count = self._count_remaining_messages(user_id=user_id, model=model)
        text = self.localization.REMAINING_MESSAGES.format(model=model, remaining_count=remaining_count)
        await message.reply(text)

    #
    # Subscription
    #

    async def sub_info(self, message: Message) -> None:
        assert message.from_user
        user_id = message.from_user.id
        remaining_seconds = self.db.get_subscription_info(user_id)
        text = self.localization.INACTIVE_SUB
        if remaining_seconds > 0:
            text = self.localization.ACTIVE_SUB.format(remaining_hours=remaining_seconds // 3600)
        await message.reply(text)

    async def sub_buy(self, message: Message) -> None:
        assert message.from_user
        user_id = message.from_user.id
        chat_id = message.chat.id
        is_chat = chat_id != user_id
        if is_chat:
            await message.reply(self.localization.SUB_NOT_CHAT)
            return

        remaining_seconds = self.db.get_subscription_info(user_id)
        if remaining_seconds > 0:
            await message.reply(self.localization.ACTIVE_SUB.format(remaining_hours=remaining_seconds // 3600))
            return

        limits = {name: provider.limits for name, provider in self.providers.items()}
        sub_limits = self.localization.LIMITS.render(limits=limits, mode="subscribed").strip()
        description = self.localization.SUB_DESCRIPTION.render(
            sub_limits=sub_limits,
            price_week=self.config.sub_configs[SubKey.RUB_WEEK].price,
            price_month=self.config.sub_configs[SubKey.RUB_MONTH].price,
        )
        await message.reply(description, parse_mode=ParseMode.MARKDOWN, reply_markup=self.buy_kb.as_markup())

    async def stars_sub_buy_proceed(self, callback: CallbackQuery) -> None:
        assert callback.from_user
        assert callback.message
        assert isinstance(callback.message, Message)
        assert callback.data
        assert "buy:stars:" in callback.data

        sub_key_str = callback.data.split(":")[2]
        user_id = callback.from_user.id
        chat_id = callback.message.chat.id
        is_chat = chat_id != user_id
        if is_chat:
            await callback.message.reply(self.localization.SUB_NOT_CHAT)
            return

        remaining_seconds = self.db.get_subscription_info(user_id)
        if remaining_seconds > 0:
            await callback.message.reply(self.localization.ACTIVE_SUB.format(remaining_hours=remaining_seconds // 3600))
            return

        key = SubKey(sub_key_str)
        key_to_short_title = {
            SubKey.XTR_WEEK: self.localization.SUB_WEEK_SHORT_TITLE,
            SubKey.XTR_MONTH: self.localization.SUB_MONTH_SHORT_TITLE,
        }
        key_to_title = {
            SubKey.XTR_WEEK: self.localization.SUB_WEEK_TITLE,
            SubKey.XTR_MONTH: self.localization.SUB_MONTH_TITLE
        }
        title = key_to_short_title[key]
        description = key_to_title[key].format(user_id=user_id)
        sub = self.config.sub_configs[key]
        await self.bot.send_invoice(
            chat_id,
            title=title,
            description=description,
            prices=[LabeledPrice(label=title, amount=sub.price)],
            provider_token="",
            currency="XTR",
            payload=f"{user_id}#{sub.duration}",
            reply_to_message_id=callback.message.message_id,
        )

    async def pre_checkout_handler(self, pre_checkout_query: PreCheckoutQuery) -> None:
        try:
            await self.bot.answer_pre_checkout_query(pre_checkout_query.id, ok=True)
        except Exception as e:
            await self.bot.answer_pre_checkout_query(pre_checkout_query.id, ok=False, error_message=str(e))

    async def successful_payment_handler(self, message: Message) -> None:
        assert message.successful_payment
        successful_payment: SuccessfulPayment = message.successful_payment
        assert message.chat
        assert message.from_user
        chat_id = message.chat.id
        user_id = message.from_user.id
        assert successful_payment
        payload = successful_payment.invoice_payload
        charge_id = successful_payment.telegram_payment_charge_id
        self.db.add_charge(user_id, charge_id)
        payload_user_id, payload_duration = payload.split("#")
        assert user_id == int(payload_user_id)
        self.db.subscribe_user(user_id, int(payload_duration))
        await self.bot.send_message(chat_id, self.localization.SUB_SUCCESS)

    async def yookassa_sub_buy_proceed(self, callback: CallbackQuery) -> None:
        assert self.yookassa
        assert callback.from_user
        assert callback.message
        assert callback.data
        assert "buy:yookassa:" in callback.data
        assert isinstance(callback.message, Message)
        assert self.bot_info
        assert self.bot_info.username

        sub_key_str = callback.data.split(":")[2]
        user_id = callback.from_user.id
        email = self.db.get_email(user_id)
        if not email:
            await callback.message.reply(self.localization.SET_EMAIL)
            return

        chat_id = callback.message.chat.id
        is_chat = chat_id != user_id
        if is_chat:
            await callback.message.reply(self.localization.SUB_NOT_CHAT)
            return

        remaining_seconds = self.db.get_subscription_info(user_id)
        if remaining_seconds > 0:
            await callback.message.reply(self.localization.ACTIVE_SUB.format(remaining_hours=remaining_seconds // 3600))
            return

        key = SubKey(sub_key_str)
        sub = self.config.sub_configs[key]
        key_to_title = {
            SubKey.RUB_WEEK: self.localization.SUB_WEEK_TITLE,
            SubKey.RUB_MONTH: self.localization.SUB_MONTH_TITLE
        }
        title = key_to_title[key].format(user_id=user_id)
        payment_data = self.yookassa.create_payment(
            sub.price, title, email=email, bot_username=self.bot_info.username
        )
        payment_id = payment_data["id"]
        timestamp = self.db.get_current_ts()
        try:
            url = payment_data["confirmation"]["confirmation_url"]
            status = payment_data["status"]
            self.db.save_payment(
                payment_id=payment_id, user_id=user_id, chat_id=chat_id, url=url, status=status, timestamp=timestamp
            )
            await callback.message.reply(self.localization.PAYMENT_URL.format(url=url))
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
                sub_key = SubKey(self.yookassa.get_sub_key(payment.payment_id))
                self.db.subscribe_user(payment.user_id, self.config.sub_configs[sub_key].duration)
                await self.bot.send_message(chat_id=payment.chat_id, text=self.localization.SUB_SUCCESS)
                self.db.set_payment_status(payment.payment_id, status=status.value, internal_status="completed")
            elif status == YookassaStatus.CANCELED:
                self.db.set_payment_status(payment.payment_id, status=status.value, internal_status="completed")
                await self.bot.send_message(chat_id=payment.chat_id, text=self.localization.PAYMENT_CANCEL)

    async def set_email(self, message: Message) -> None:
        assert message.text
        assert message.from_user
        email = message.text.replace("/setemail", "").strip()
        is_valid = "@" in parseaddr(email)[1]
        if not is_valid:
            await message.reply(self.localization.INCORRECT_EMAIL)
            return
        self.db.set_email(message.from_user.id, email)
        await message.reply(self.localization.FILLED_EMAIL.format(email=email))

    #
    # Generation parameters
    #

    @check_admin
    async def set_temperature(self, message: Message) -> None:
        await message.reply(self.localization.SELECT_TEMPERATURE, reply_markup=self.temperature_kb.as_markup())

    @check_admin
    async def set_temperature_button_handler(self, callback: CallbackQuery) -> None:
        assert callback.message
        assert callback.data
        chat_id = callback.message.chat.id
        model = self.db.get_current_model(chat_id)
        provider = self.providers[model]
        temperature = float(callback.data.split(":")[1])
        params = self.db.get_parameters(chat_id)
        params = copy.deepcopy(provider.params) if params is None else params
        params["temperature"] = temperature
        self.db.set_parameters(chat_id, **params)
        assert isinstance(callback.message, Message)
        await callback.message.edit_text(self.localization.NEW_TEMPERATURE.format(temperature=temperature))

    @check_admin
    async def set_top_p(self, message: Message) -> None:
        await message.reply(self.localization.SELECT_TOP_P, reply_markup=self.top_p_kb.as_markup())

    @check_admin
    async def set_top_p_button_handler(self, callback: CallbackQuery) -> None:
        assert callback.message
        assert callback.data
        chat_id = callback.message.chat.id
        model = self.db.get_current_model(chat_id)
        provider = self.providers[model]
        top_p = float(callback.data.split(":")[1])
        params = self.db.get_parameters(chat_id)
        params = provider.params if params is None else params
        params["top_p"] = top_p
        self.db.set_parameters(chat_id, **params)
        assert isinstance(callback.message, Message)
        await callback.message.edit_text(self.localization.NEW_TOP_P.format(top_p=top_p))

    @check_admin
    async def set_frequency_penalty(self, message: Message) -> None:
        await message.reply(self.localization.SELECT_FREQUENCY_PENALTY, reply_markup=self.freq_penalty_kb.as_markup())

    @check_admin
    async def set_frequency_penalty_button_handler(self, callback: CallbackQuery) -> None:
        assert callback.message
        assert callback.data
        chat_id = callback.message.chat.id
        model = self.db.get_current_model(chat_id)
        provider = self.providers[model]
        frequency_penalty = float(callback.data.split(":")[1])
        params = self.db.get_parameters(chat_id)
        params = provider.params if params is None else params
        params["frequency_penalty"] = frequency_penalty
        self.db.set_parameters(chat_id, **params)
        assert isinstance(callback.message, Message)
        await callback.message.edit_text(
            self.localization.NEW_FREQUENCY_PENALTY.format(frequency_penalty=frequency_penalty)
        )

    async def get_params(self, message: Message) -> None:
        chat_id = message.chat.id
        model = self.db.get_current_model(chat_id)
        provider = self.providers[model]
        params = self.db.get_parameters(chat_id)
        params = provider.params if params is None else params
        params_str = json.dumps(params)
        await message.reply(self.localization.CURRENT_PARAMS.format(params=params_str))

    #
    # Tools
    #

    def _get_tools(self, chat_id: int) -> Optional[List[Dict[str, Any]]]:
        model = self.db.get_current_model(chat_id)
        provider = self.providers[model]
        if provider.can_handle_tools and self.tools and self.db.are_tools_enabled(chat_id):
            return [t.get_specification() for t in self.tools.values()]
        return None

    @check_admin
    async def toogle_tools(self, message: Message) -> None:
        chat_id = message.chat.id
        model = self.db.get_current_model(chat_id)
        provider = self.providers[model]
        if not provider.can_handle_tools:
            await message.reply(self.localization.TOOLS_NOT_SUPPORTED_BY_MODEL.format(model=model))
            return
        current_value = self.db.are_tools_enabled(chat_id)
        self.db.set_enable_tools(chat_id, not current_value)
        self.db.create_conv_id(chat_id)
        if not current_value:
            await message.reply(self.localization.ENABLED_TOOLS)
        else:
            await message.reply(self.localization.DISABLED_TOOLS)

    async def _check_tools(self, messages: ChatMessages, model: str) -> Any:
        messages = copy.deepcopy(messages)
        messages = self._replace_images(messages)
        messages = self._fix_broken_tool_calls(messages)
        tools = [t.get_specification() for t in self.tools.values()]
        casted_messages = [cast(ChatCompletionMessageParam, message) for message in messages]
        casted_tools = [cast(ChatCompletionToolParam, tool) for tool in tools]
        chat_completion = await self.providers[model].api.chat.completions.create(
            model=self.providers[model].model_name,
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
            await placeholder.edit_text(self.localization.DALLE_LIMIT)
            return

        displayed_prompt = kwargs["prompt_russian"]
        await placeholder.edit_text(self.localization.DALLE_PROMPT.format(displayed_prompt=displayed_prompt))
        function_response = await self.tools["dalle"](**kwargs)
        if not isinstance(function_response, list) or "image_url" not in function_response[1]:
            text = self.localization.DALLE_ERROR.format(error=function_response)
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
        base64_image = image_url["url"].split(",")[-1]
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
                print(f"Malformed tool arguments: {tool_call.function.arguments}")

            if function_response is None and function_name == "dalle":
                await self._call_dalle(
                    chat_id=chat_id, user_id=user_id, conv_id=conv_id, placeholder=placeholder, **function_args
                )
                return None

            if function_response is None:
                try:
                    function_response = await function_to_call(**function_args)
                except Exception as e:
                    function_response = f"Tool call error: {str(e)}"

            assert function_response is not None
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
    # Text generation
    #

    async def generate(self, message: Message) -> None:
        assert message.from_user
        user_id = message.from_user.id
        user_name = self._get_user_name(message.from_user)
        is_chat = message.chat.type in ("group", "supergroup")
        chat_id = message.chat.id if is_chat else user_id
        if is_chat:
            chat_id = message.chat.id
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

        await self._handle_message(message)

    async def _handle_message(self, message: Message, override_content: Optional[str] = None) -> None:
        assert message.from_user
        user_id = message.from_user.id
        user_name = self._get_user_name(message.from_user)
        is_chat = message.chat.type in ("group", "supergroup")
        chat_id = message.chat.id if is_chat else user_id
        model = self.db.get_current_model(chat_id)

        if model not in self.providers:
            await message.reply(self.localization.MODEL_NOT_SUPPORTED)
            return
        provider = self.providers[model]

        remaining_count = self._count_remaining_messages(user_id=user_id, model=model)
        print(user_id, model, remaining_count)
        if remaining_count <= 0:
            await message.reply(self.localization.LIMIT_EXCEEDED.format(model=model))
            return

        is_big_file = await self._is_big_file(message)
        if is_big_file:
            await message.reply(self.localization.FILE_IS_TOO_BIG)
            return

        conv_id = self.db.get_current_conv_id(chat_id)
        history = self.db.fetch_conversation(conv_id)
        params = self.db.get_parameters(chat_id)
        params = provider.params if params is None else params
        params = copy.deepcopy(params)
        system_prompt = self.db.get_system_prompt(chat_id)
        system_prompt = provider.system_prompt if system_prompt is None else system_prompt
        content = await self._build_content(message)

        if "claude" in model and params["temperature"] > 1.0:
            await message.reply(self.localization.CLAUDE_HIGH_TEMPERATURE)
            return
        if "o1" in model and (params["temperature"] != 1.0 or params["top_p"] != 1.0 or system_prompt):
            await message.reply(self.localization.O1_WRONG_PARAMS)
            return
        if content is None:
            await message.reply(self.localization.CONTENT_NOT_SUPPORTED)
            return
        if not isinstance(content, str) and not provider.can_handle_images:
            await message.reply(self.localization.CONTENT_NOT_SUPPORTED_BY_MODEL)
            return

        self.db.save_user_message(content, conv_id=conv_id, user_id=user_id, user_name=user_name)

        history = history + [{"role": "user", "content": content, "user_name": user_name}]
        history = self._prepare_history(history, provider, is_chat)

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
            history = self._merge_messages(history)
            if tools and "gpt" not in model:
                params["tools"] = tools
            answer = await self._query_api(provider=provider, messages=history, system_prompt=system_prompt, **params)

            if is_chat and self.llm_filter:
                all_messages = history + [{"role": "assistant", "content": answer}]
                filter_result = await self.llm_filter(all_messages)
                if filter_result:
                    answer = self.localization.MESSAGE_FILTERED

            output_chunk_size = self.config.output_chunk_size
            if output_chunk_size is not None:
                answer_parts = _split_message(answer, output_chunk_size=output_chunk_size)
            else:
                answer_parts = [answer]

            new_message = await _edit_text(placeholder, answer_parts[0])
            assert isinstance(new_message, Message)
            for part in answer_parts[1:]:
                new_message = await _reply(message, part)
                assert isinstance(new_message, Message)

            markup = self.likes_kb.as_markup()
            new_message = await _edit_text(
                new_message,
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
            text = self.localization.ERROR.format(admin_username=self.config.admin_user_name, conv_id=conv_id)
            await placeholder.edit_text(text)

    @check_creator
    async def debug(self, message: Message, command: CommandObject) -> None:
        assert command.args
        conv_id = command.args.strip()
        user_id = self.db.get_user_id_by_conv_id(conv_id)
        history = self.db.fetch_conversation(conv_id)
        model = list({m["model"] for m in history if m["model"]})[0]
        provider = self.providers[model]
        history = self._prepare_history(history, provider, False)
        params = self.db.get_parameters(user_id)
        params = provider.params if params is None else params
        params = copy.deepcopy(params)
        system_prompt = self.db.get_system_prompt(user_id)
        system_prompt = provider.system_prompt if system_prompt is None else system_prompt
        placeholder = await message.reply("💬")
        history = self._fix_image_roles(history)
        history = self._fix_broken_tool_calls(history)
        history = self._merge_messages(history)
        if history[-1]["role"] == "assistant":
            history.pop()
        answer = await self._query_api(provider=provider, messages=history, system_prompt=system_prompt, **params)
        await placeholder.edit_text(answer[: self.config.output_chunk_size])

    @staticmethod
    async def _query_api(
        provider: LLMProvider,
        messages: ChatMessages,
        system_prompt: str,
        num_retries: int = 2,
        **kwargs: Any
    ) -> str:
        assert messages
        if messages[0]["role"] != "system" and system_prompt.strip():
            messages.insert(0, {"role": "system", "content": system_prompt})

        if provider.merge_system and messages[0]["role"] == "system":
            system_message = messages[0]["content"]
            messages = messages[1:]
            messages[0]["content"] = system_message + "\n\n" + messages[0]["content"]

        print(
            provider.provider_name,
            "####",
            len(messages),
            "####",
            _crop_content(messages[-1]["content"]),
        )
        casted_messages = [cast(ChatCompletionMessageParam, message) for message in messages]
        answer: Optional[str] = None
        for _ in range(num_retries):
            try:
                chat_completion = await provider.api.chat.completions.create(
                    model=provider.model_name, messages=casted_messages, **kwargs
                )
                assert chat_completion.choices, str(chat_completion)
                assert chat_completion.choices[0].message.content, str(chat_completion)
                assert isinstance(chat_completion.choices[0].message.content, str), str(chat_completion)
                answer = chat_completion.choices[0].message.content
                break
            except Exception:
                traceback.print_exc()
                continue
        assert answer
        if provider.merge_spaces:
            answer = answer.replace("  ", " ")
        print(
            provider.provider_name,
            "####",
            len(messages),
            "####",
            _crop_content(messages[-1]["content"]),
            "####",
            _crop_content(answer),
        )
        return answer

    async def _build_content(self, message: Message) -> Union[None, str, List[Dict[str, Any]]]:
        content_type = message.content_type
        chat_id = message.chat.id
        bot_short_name = self.db.get_short_name(chat_id)
        if content_type == "text":
            assert message.text
            text = message.text
            assert self.bot_info
            assert self.bot_info.username
            text = text.replace("@" + self.bot_info.username, bot_short_name).strip()
            return text

        photo = None
        text_document = None
        photo_ext = (".jpg", "jpeg", ".png", ".webp", ".gif")
        if content_type == "photo":
            assert message.photo
            file_info = await self.bot.get_file(message.photo[-1].file_id)
            photo = file_info.file_path
        elif content_type == "document":
            document = message.document
            if document:
                try:
                    file_info = await self.bot.get_file(document.file_id)
                except TelegramBadRequest:
                    traceback.print_exc()
                    file_info = None
                if file_info and file_info.file_path:
                    file_path = file_info.file_path
                    file_ext = "." + file_path.split(".")[-1].lower()
                    if file_ext in photo_ext:
                        photo = file_path
                    if self.document_loader.is_supported(file_ext):
                        text_document = file_path

        if photo:
            image_file_stream: Optional[BinaryIO] = await self.bot.download_file(photo)
            assert image_file_stream
            base64_image = base64.b64encode(image_file_stream.read()).decode("utf-8")
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

        if text_document:
            file_stream: Optional[BinaryIO] = await self.bot.download_file(text_document)
            assert file_stream
            extracted_text = self.document_loader.load(file_stream, file_ext)
            if extracted_text:
                caption = message.caption if message.caption else ""
                if caption:
                    extracted_text = f"{caption}\n\n#####\n{extracted_text}\n#####\n\n{caption}"
                return extracted_text

        return None

    async def _is_big_file(self, message: Message) -> bool:
        content_type = message.content_type
        if content_type != "document":
            return False
        document = message.document
        if not document:
            return False
        try:
            await self.bot.get_file(document.file_id)
        except TelegramBadRequest:
            if "file is too big" in traceback.format_exc():
                return True
        return False

    #
    # Auxiliary methods
    #

    async def save_feedback_handler(self, callback: CallbackQuery) -> None:
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

    def _count_tokens(self, messages: ChatMessages, provider: LLMProvider) -> int:
        tokenizer_name = provider.tokenizer_name
        if not tokenizer_name:
            tokenizer_name = provider.model_name
        url = str(provider.api.base_url)
        tokens_count = 0

        if "api.openai.com" in url:
            if "o1" in tokenizer_name:
                return 0
            encoding = tiktoken.encoding_for_model(tokenizer_name)
            for m in messages:
                if isinstance(m["content"], str):
                    tokens_count += len(encoding.encode(m["content"]))
                elif self._is_image_content(m["content"]):
                    tokens_count += 2000
            return tokens_count

        if "anthropic" in url:
            for m in messages:
                if isinstance(m["content"], str):
                    tokens_count += len(m["content"]) // 2
            return tokens_count

        tokenizer = Tokenizers.get(tokenizer_name)
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

    def _get_user_name(self, user: User) -> str:
        return str(user.full_name) if user.full_name else str(user.username)

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
                texts = []
                for item in m["content"]:
                    if "text" in item:
                        texts.append(item["text"])
                m["content"] = IMAGE_PLACEHOLDER + "\n" + "\n".join(texts)
        return messages

    def _truncate_text(self, text: str) -> str:
        if self.config.output_chunk_size and len(text) > self.config.output_chunk_size:
            text = text[: self.config.output_chunk_size] + "... truncated"
        return text


def main(
    bot_config_path: str,
    providers_config_path: str,
    db_path: str,
    localization_config_path: str,
    characters_path: Optional[str] = None,
    tools_config_path: Optional[str] = None,
    yookassa_config_path: Optional[str] = None,
) -> None:
    bot = LlmBot(
        bot_config_path=bot_config_path,
        providers_config_path=providers_config_path,
        db_path=db_path,
        characters_path=characters_path,
        tools_config_path=tools_config_path,
        yookassa_config_path=yookassa_config_path,
        localization_config_path=localization_config_path,
    )
    asyncio.run(bot.start_polling())


if __name__ == "__main__":
    fire.Fire(main)
