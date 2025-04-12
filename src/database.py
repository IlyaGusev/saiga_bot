import secrets
import json
from typing import Optional, List, Any, Dict, Union
from datetime import datetime, timezone

from sqlalchemy import create_engine, Integer, String, Text, MetaData, func
from sqlalchemy.orm import DeclarativeBase, sessionmaker, mapped_column, Mapped


metadata = MetaData()

DEFAULT_SHORT_NAME = "Сайга"
DEFAULT_MODEL = "saiga-nemo-12b"
DEFAULT_PARAMS = {
    "temperature": 0.6,
    "top_p": 0.9,
    "max_tokens": 1536,
}


class Base(DeclarativeBase):
    pass


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    role: Mapped[str]
    user_id: Mapped[Optional[int]]
    user_name: Mapped[Optional[str]]
    reply_user_id: Mapped[Optional[int]]
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    conv_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    timestamp: Mapped[Optional[int]]
    message_id: Mapped[Optional[int]]
    model: Mapped[Optional[str]]
    system_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tool_calls: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tool_call_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class Conversation(Base):
    __tablename__ = "current_conversations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    conv_id: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    timestamp: Mapped[int]


class Model(Base):
    __tablename__ = "models"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, unique=True, index=True)
    model: Mapped[str]


class ModelParameters(Base):
    __tablename__ = "model_parameters"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    model: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    short_name: Mapped[Optional[str]]
    generation_parameters: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    enable_tools: Mapped[Optional[bool]]


class Like(Base):
    __tablename__ = "likes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    message_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    feedback: Mapped[str]
    is_correct: Mapped[int]


class Subscription(Base):
    __tablename__ = "subscriptions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    from_timestamp: Mapped[int]
    until_timestamp: Mapped[int]


class Payment(Base):
    __tablename__ = "Payments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[int]
    payment_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    chat_id: Mapped[int]
    status: Mapped[str]
    internal_status: Mapped[str]
    url: Mapped[Optional[str]]


class Email(Base):
    __tablename__ = "Emails"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    email: Mapped[str]


class Charge(Base):
    __tablename__ = "charges"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int]
    timestamp: Mapped[int]
    charge_id: Mapped[str]


class Database:
    def __init__(self, db_path: str):
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    @staticmethod
    def get_current_ts() -> int:
        return int(datetime.now().replace(tzinfo=timezone.utc).timestamp())

    def set_email(self, user_id: int, email: str) -> None:
        with self.Session() as session:
            obj = session.query(Email).filter(Email.user_id == user_id).first()
            if obj:
                obj.email = email
            else:
                session.add(Email(user_id=user_id, email=email))
            session.commit()

    def add_charge(self, user_id: int, charge_id: str) -> None:
        with self.Session() as session:
            timestamp = self.get_current_ts()
            session.add(Charge(user_id=user_id, charge_id=charge_id, timestamp=timestamp))
            session.commit()

    def get_email(self, user_id: int) -> Optional[str]:
        with self.Session() as session:
            obj = session.query(Email).filter(Email.user_id == user_id).first()
            return obj.email if obj else None

    def save_payment(
        self,
        payment_id: str,
        user_id: int,
        chat_id: int,
        status: str,
        url: str,
        timestamp: int,
    ) -> None:
        with self.Session() as session:
            new_payment = Payment(
                payment_id=payment_id,
                user_id=user_id,
                chat_id=chat_id,
                status=status,
                internal_status="waiting",
                url=url,
                timestamp=timestamp,
            )
            session.add(new_payment)
            session.commit()

    def get_waiting_payments(self) -> List[Payment]:
        with self.Session() as session:
            return session.query(Payment).filter(Payment.internal_status == "waiting").all()

    def set_payment_status(self, payment_id: str, status: str, internal_status: str) -> None:
        with self.Session() as session:
            payment = session.query(Payment).filter(Payment.payment_id == payment_id).first()
            if payment:
                payment.status = status
                payment.internal_status = internal_status
                session.commit()

    def create_conv_id(self, user_id: int) -> str:
        conv_id = secrets.token_hex(nbytes=16)
        with self.Session() as session:
            new_conv = Conversation(user_id=user_id, conv_id=conv_id, timestamp=self.get_current_ts())
            session.add(new_conv)
            session.commit()
        return conv_id

    def get_user_id_by_conv_id(self, conv_id: str) -> int:
        with self.Session() as session:
            conv = (
                session.query(Conversation)
                .filter(Conversation.conv_id == conv_id)
                .order_by(Conversation.timestamp.desc())
                .first()
            )
            assert conv
            return conv.user_id

    def get_current_conv_id(self, user_id: int) -> str:
        with self.Session() as session:
            conv = (
                session.query(Conversation)
                .filter(Conversation.user_id == user_id)
                .order_by(Conversation.timestamp.desc())
                .first()
            )
            return conv.conv_id if conv else self.create_conv_id(user_id)

    def fetch_conversation(self, conv_id: str) -> List[Any]:
        with self.Session() as session:
            messages = session.query(Message).filter(Message.conv_id == conv_id).order_by(Message.timestamp).all()
            if not messages:
                return []
            clean_messages = []
            for m in messages:
                message = {
                    "role": m.role,
                    "content": self._parse_content(m.content),
                    "model": m.model,
                    "system_prompt": m.system_prompt,
                    "timestamp": m.timestamp,
                    "user_id": m.user_id,
                    "user_name": m.user_name,
                    "tool_calls": self._parse_content(m.tool_calls),
                    "tool_call_id": m.tool_call_id,
                    "name": m.name,
                }
                clean_messages.append(message)
            return clean_messages

    def get_user_id(self, user_name: str) -> int:
        with self.Session() as session:
            user_id = session.query(Message.user_id).filter(Message.user_name == user_name).distinct().first()
            assert user_id, f"User ID not found for {user_name}"
            return int(user_id[0])

    def get_current_model(self, user_id: int) -> str:
        with self.Session() as session:
            model = session.query(Model).filter(Model.user_id == user_id).first()
            return model.model if model else DEFAULT_MODEL

    def set_current_model(self, user_id: int, model_name: str) -> None:
        with self.Session() as session:
            model = session.query(Model).filter(Model.user_id == user_id).first()
            if model:
                model.model = model_name
            else:
                session.add(Model(user_id=user_id, model=model_name))
            session.commit()

    def get_current_model_parameters(self, user_id: int) -> Optional[ModelParameters]:
        current_model = self.get_current_model(user_id)
        with self.Session() as session:
            return session.query(ModelParameters).filter_by(user_id=user_id, model=current_model).first()

    def get_system_prompt(self, user_id: int) -> Optional[str]:
        params = self.get_current_model_parameters(user_id)
        if params and params.prompt is not None:
            return params.prompt
        return None

    def set_system_prompt(self, user_id: int, text: str) -> None:
        current_model = self.get_current_model(user_id)
        with self.Session() as session:
            params = session.query(ModelParameters).filter_by(user_id=user_id, model=current_model).first()
            if params:
                params.prompt = text
            else:
                session.add(ModelParameters(user_id=user_id, model=current_model, prompt=text))
            session.commit()

    def get_short_name(self, user_id: int) -> str:
        params = self.get_current_model_parameters(user_id)
        return params.short_name if params and params.short_name else DEFAULT_SHORT_NAME

    def set_short_name(self, user_id: int, text: str) -> None:
        current_model = self.get_current_model(user_id)
        with self.Session() as session:
            params = session.query(ModelParameters).filter_by(user_id=user_id, model=current_model).first()
            if params:
                params.short_name = text
            else:
                session.add(ModelParameters(user_id=user_id, model=current_model, short_name=text))
            session.commit()

    def set_parameters(self, user_id: int, **kwargs: Any) -> None:
        current_model = self.get_current_model(user_id)
        generation_parameters = self.get_parameters(user_id)
        if generation_parameters is not None:
            generation_parameters.update(kwargs)
        else:
            generation_parameters = kwargs
        with self.Session() as session:
            params = session.query(ModelParameters).filter_by(user_id=user_id, model=current_model).first()
            if params:
                params.generation_parameters = json.dumps(generation_parameters)
            else:
                session.add(
                    ModelParameters(
                        user_id=user_id,
                        model=current_model,
                        generation_parameters=json.dumps(params),
                    )
                )
            session.commit()

    def get_parameters(self, user_id: int) -> Optional[Dict[str, Any]]:
        params = self.get_current_model_parameters(user_id)
        if params and params.generation_parameters and params.generation_parameters != "null":
            parsed_params: Dict[str, Any] = json.loads(params.generation_parameters)
            parsed_params.pop("tools", None)
            return parsed_params
        return None

    def are_tools_enabled(self, user_id: int) -> bool:
        params = self.get_current_model_parameters(user_id)
        return params.enable_tools if params and params.enable_tools is not None else False

    def set_enable_tools(self, user_id: int, value: bool) -> None:
        current_model = self.get_current_model(user_id)
        with self.Session() as session:
            params = session.query(ModelParameters).filter_by(user_id=user_id, model=current_model).first()
            if params:
                params.enable_tools = value
            else:
                session.add(ModelParameters(user_id=user_id, model=current_model, enable_tools=value))
            session.commit()

    def save_user_message(
        self,
        content: Union[None, str, List[Dict[str, Any]]],
        conv_id: str,
        user_id: int,
        user_name: Optional[str] = None,
    ) -> None:
        with self.Session() as session:
            new_message = Message(
                role="user",
                content=self._serialize_content(content),
                conv_id=conv_id,
                user_id=user_id,
                user_name=user_name,
                timestamp=self.get_current_ts(),
            )
            session.add(new_message)
            session.commit()

    def save_assistant_message(
        self,
        content: Union[str, List[Dict[str, Any]]],
        conv_id: str,
        message_id: int,
        model: str,
        system_prompt: Optional[str] = None,
        reply_user_id: Optional[int] = None,
    ) -> None:
        with self.Session() as session:
            new_message = Message(
                role="assistant",
                content=self._serialize_content(content),
                conv_id=conv_id,
                timestamp=self.get_current_ts(),
                message_id=message_id,
                model=model,
                system_prompt=system_prompt,
                reply_user_id=reply_user_id,
            )
            session.add(new_message)
            session.commit()

    def save_tool_answer_message(
        self,
        content: Union[str, List[Dict[str, Any]]],
        conv_id: str,
        model: str,
        tool_call_id: str,
        name: str,
    ) -> None:
        with self.Session() as session:
            new_message = Message(
                role="tool",
                content=self._serialize_content(content),
                conv_id=conv_id,
                timestamp=self.get_current_ts(),
                model=model,
                tool_call_id=tool_call_id,
                name=name,
            )
            session.add(new_message)
            session.commit()

    def save_tool_calls_message(
        self,
        tool_calls: Any,
        conv_id: str,
        model: str,
    ) -> None:
        with self.Session() as session:
            new_message = Message(
                role="assistant",
                content=None,
                conv_id=conv_id,
                timestamp=self.get_current_ts(),
                model=model,
                tool_calls=self._serialize_content(tool_calls),
            )
            session.add(new_message)
            session.commit()

    def save_feedback(self, feedback: str, user_id: int, message_id: int) -> None:
        with self.Session() as session:
            new_feedback = Like(
                user_id=user_id,
                message_id=message_id,
                feedback=feedback,
                is_correct=True,
            )
            session.add(new_feedback)
            session.commit()

    def count_user_messages(self, user_id: int, model: str, interval: int) -> int:
        with self.Session() as session:
            current_ts = self.get_current_ts()
            count = (
                session.query(func.count(Message.id))
                .filter(
                    Message.reply_user_id == user_id,
                    Message.role == "assistant",
                    Message.model == model,
                    Message.timestamp.isnot(None),
                    Message.timestamp > (current_ts - interval),
                )
                .scalar()
            )
            return int(count)

    def count_generated_images(self, user_id: int, interval: int) -> int:
        with self.Session() as session:
            current_ts = self.get_current_ts()
            count = (
                session.query(func.count(Message.id))
                .filter(
                    Message.reply_user_id == user_id,
                    Message.role == "assistant",
                    Message.model == "dalle",
                    Message.timestamp.isnot(None),
                    Message.timestamp > (current_ts - interval),
                )
                .scalar()
            )
            return int(count)

    def is_subscribed_user(self, user_id: int) -> bool:
        remaining_time = self.get_subscription_info(user_id)
        return remaining_time > 0

    def get_subscription_info(self, user_id: int) -> int:
        with self.Session() as session:
            subscriptions = (session.query(Subscription).filter(Subscription.user_id == user_id)).all()
            if not subscriptions:
                return 0
            until_timestamp = max(subscriptions, key=lambda x: x.until_timestamp).until_timestamp
            remaining_time = until_timestamp - self.get_current_ts()
            return max(remaining_time, 0)

    def subscribe_user(self, user_id: int, duration: int) -> None:
        with self.Session() as session:
            current_ts = self.get_current_ts()
            new_subscription = Subscription(
                user_id=user_id,
                from_timestamp=current_ts,
                until_timestamp=current_ts + duration,
            )
            session.add(new_subscription)
            session.commit()

    def get_all_conv_ids(self, min_timestamp: Optional[int] = None) -> List[str]:
        with self.Session() as session:
            if min_timestamp is None:
                conversations = session.query(Conversation).all()
            else:
                conversations = session.query(Conversation).filter(Conversation.timestamp >= min_timestamp).all()
            return [conv.conv_id for conv in conversations]

    def _serialize_content(self, content: Union[None, str, List[Dict[str, Any]]]) -> str:
        if isinstance(content, str):
            return content
        return json.dumps(content)

    def _parse_content(self, content: Any) -> Any:
        try:
            if content is None:
                return None
            parsed_content = json.loads(content)
            if not isinstance(parsed_content, list):
                return content
            for m in parsed_content:
                if not isinstance(m, dict):
                    return content
            return parsed_content
        except json.JSONDecodeError:
            return content
