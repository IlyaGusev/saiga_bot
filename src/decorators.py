from typing import Callable, Coroutine, Any, Union
from functools import wraps

from aiogram.types import (
    Message,
    CallbackQuery,
)


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
