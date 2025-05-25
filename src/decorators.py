from typing import Callable, Coroutine, Any, Union, Optional, TypeVar, Type, Dict
from functools import wraps

from aiogram.types import (
    Message,
    CallbackQuery,
)
from smolagents import Tool  # type: ignore

from src.database import Database


def check_admin(
    func: Callable[..., Coroutine[Any, Any, Any]],
) -> Callable[..., Coroutine[Any, Any, Any]]:
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
            username = self._get_user_name(user)
            is_admin = await self._is_admin(user_id=user_id, chat_id=chat_id)
            if not is_admin:
                await self.bot.send_message(
                    chat_id=chat_id,
                    text=self.localization.ADMINS_ONLY.format(username=username),
                )
                return
        return await func(self, obj, *args, **kwargs)

    return wrapped


def check_creator(
    func: Callable[..., Coroutine[Any, Any, Any]],
) -> Callable[..., Coroutine[Any, Any, Any]]:
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

        if user_id != self.config.admin_user_id:
            username = self._get_user_name(user)
            await self.bot.send_message(
                chat_id=chat_id,
                text=self.localization.CREATORS_ONLY.format(username=username),
            )
            return
        return await func(self, obj, *args, **kwargs)

    return wrapped


ToolClass = TypeVar("ToolClass", bound=Type[Tool])


def log_tool_call(cls: ToolClass) -> ToolClass:
    original_init = cls.__init__

    @wraps(original_init)
    def new_init(
        self: Any,
        *args: Any,
        user_id: Optional[int] = None,
        db: Optional[Database] = None,
        limits: Optional[Dict[str, Dict[str, int]]] = None,
        **kwargs: Any,
    ) -> None:
        original_init(self, *args, **kwargs)

        self.user_id = user_id
        self.db = db
        self.limits = limits

        original_forward = self.forward

        @wraps(original_forward)
        def wrapped_forward(*args: Any, **kwargs: Any) -> Any:
            if self.db is None:
                return original_forward(*args, **kwargs)
            if self.limits:
                mode = "standard" if self.db.get_subscription_info(self.user_id) <= 0 else "subscribed"
                limits = self.limits[mode]
                limit, interval = limits.limit, limits.interval
                count = self.db.count_tool_calls(user_id=self.user_id, tool_name=self.name, interval=interval)
                remaining_count = limit - count
                if remaining_count <= 0:
                    return "Error: The tool's quota for this user has been exhausted."
            result = original_forward(*args, **kwargs)
            self.db.save_tool_call(tool_name=self.name, user_id=self.user_id)
            return result

        self.forward = wrapped_forward

    cls.__init__ = new_init
    return cls
