import secrets
import json
import copy
from datetime import datetime, timezone

from sqlalchemy import create_engine, Column, Integer, String, Text, MetaData, func
from sqlalchemy.orm import declarative_base, sessionmaker


Base = declarative_base()
metadata = MetaData()

DEFAULT_SHORT_NAME = "Сайга"
DEFAULT_MODEL = "saiga-v6"
DEFAULT_PARAMS = {
    "temperature": 0.6,
    "top_p": 0.9,
    "max_tokens": 1536,
}


class Messages(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    role = Column(String, nullable=False)
    user_id = Column(Integer, nullable=True)
    user_name = Column(String, nullable=True)
    reply_user_id = Column(Integer, nullable=True)
    content = Column(Text, nullable=False)
    conv_id = Column(String, nullable=False, index=True)
    timestamp = Column(Integer, nullable=False)
    message_id = Column(Integer)
    model = Column(String)
    system_prompt = Column(Text)


class Conversations(Base):
    __tablename__ = "current_conversations"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False, index=True)
    conv_id = Column(String, nullable=False, unique=True)
    timestamp = Column(Integer, nullable=False)


class SystemPrompts(Base):
    __tablename__ = "system_prompts"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False, index=True)
    model = Column(String, nullable=True, index=True)
    prompt = Column(Text, nullable=False)


class GenerationParameters(Base):
    __tablename__ = "generation_parameters"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False, index=True)
    model = Column(String, nullable=True, index=True)
    parameters = Column(Text, nullable=False)


class ShortNames(Base):
    __tablename__ = "short_names"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False, index=True)
    short_name = Column(String)


class Models(Base):
    __tablename__ = "models"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False, unique=True, index=True)
    model = Column(String, nullable=False)


class Likes(Base):
    __tablename__ = "likes"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False, index=True)
    message_id = Column(Integer, nullable=False, index=True)
    feedback = Column(String, nullable=False)
    is_correct = Column(Integer, nullable=False)


class Database:
    def __init__(self, db_path: str):
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    @staticmethod
    def get_current_ts():
        return int(datetime.now().replace(tzinfo=timezone.utc).timestamp())

    def create_conv_id(self, user_id):
        conv_id = secrets.token_hex(nbytes=16)
        with self.Session() as session:
            new_conv = Conversations(
                user_id=user_id, conv_id=conv_id, timestamp=self.get_current_ts()
            )
            session.add(new_conv)
            session.commit()
        return conv_id

    def get_current_conv_id(self, user_id):
        with self.Session() as session:
            conv = (
                session.query(Conversations)
                .filter(Conversations.user_id == user_id)
                .order_by(Conversations.timestamp.desc())
                .first()
            )
            if conv is None:
                return self.create_conv_id(user_id)
            return conv.conv_id

    def fetch_conversation(self, conv_id):
        with self.Session() as session:
            messages = (
                session.query(Messages)
                .filter(Messages.conv_id == conv_id)
                .order_by(Messages.timestamp)
                .all()
            )
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
                }
                clean_messages.append(message)
            return clean_messages

    def get_current_model(self, user_id):
        with self.Session() as session:
            model = session.query(Models).filter(Models.user_id == user_id).first()
            if model:
                return model.model
            return DEFAULT_MODEL

    def set_current_model(self, user_id: int, model_name: str):
        with self.Session() as session:
            model = session.query(Models).filter(Models.user_id == user_id).first()
            if model:
                model.model = model_name
            else:
                new_model = Models(user_id=user_id, model=model_name)
                session.add(new_model)
            session.commit()

    def get_system_prompt(self, user_id, default_prompts):
        current_model = self.get_current_model(user_id)
        with self.Session() as session:
            prompt = (
                session.query(SystemPrompts)
                .filter(SystemPrompts.user_id == user_id)
                .filter(SystemPrompts.model == current_model)
                .first()
            )
            if prompt:
                return prompt.prompt
            return default_prompts.get(current_model, "")

    def set_system_prompt(self, user_id: int, text: str):
        current_model = self.get_current_model(user_id)
        with self.Session() as session:
            prompt = (
                session.query(SystemPrompts)
                .filter(SystemPrompts.user_id == user_id)
                .filter(SystemPrompts.model == current_model)
                .first()
            )
            if prompt:
                prompt.prompt = text
            else:
                new_prompt = SystemPrompts(
                    user_id=user_id, prompt=text, model=current_model
                )
                session.add(new_prompt)
            session.commit()

    def get_short_name(self, user_id: int):
        with self.Session() as session:
            name = (
                session.query(ShortNames)
                .filter(ShortNames.user_id == user_id)
                .first()
            )
            if name:
                return name.short_name
            return DEFAULT_SHORT_NAME

    def set_short_name(self, user_id: int, text: str):
        with self.Session() as session:
            name = (
                session.query(ShortNames)
                .filter(ShortNames.user_id == user_id)
                .first()
            )
            if name:
                name.short_name = text
            else:
                new_name = ShortNames(
                    user_id=user_id, short_name=text
                )
                session.add(new_name)
            session.commit()

    def set_parameters(self, user_id: int, default_params, **kwargs):
        current_model = self.get_current_model(user_id)
        params = self.get_parameters(user_id, default_params)
        for key, value in kwargs.items():
            params[key] = value
        with self.Session() as session:
            parameters = (
                session.query(GenerationParameters)
                .filter(GenerationParameters.user_id == user_id)
                .filter(GenerationParameters.model == current_model)
                .first()
            )
            if parameters:
                parameters.parameters = json.dumps(params)
            else:
                parameters = GenerationParameters(
                    user_id=user_id, model=current_model, parameters=json.dumps(params)
                )
                session.add(parameters)
            session.commit()

    def get_parameters(self, user_id: int, default_params):
        current_model = self.get_current_model(user_id)
        with self.Session() as session:
            parameters = (
                session.query(GenerationParameters)
                .filter(GenerationParameters.user_id == user_id)
                .filter(GenerationParameters.model == current_model)
                .first()
            )
            if parameters:
                return json.loads(parameters.parameters)
            return copy.deepcopy(default_params.get(current_model, DEFAULT_PARAMS))

    def save_user_message(
        self, content: str, conv_id: str, user_id: int, user_name: str = None
    ):
        with self.Session() as session:
            new_message = Messages(
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
        content: str,
        conv_id: str,
        message_id: int,
        model: str,
        system_prompt: str,
        reply_user_id: int
    ):
        with self.Session() as session:
            new_message = Messages(
                role="assistant",
                content=content,
                conv_id=conv_id,
                timestamp=self.get_current_ts(),
                message_id=message_id,
                model=model,
                system_prompt=system_prompt,
                reply_user_id=reply_user_id
            )
            session.add(new_message)
            session.commit()

    def save_feedback(self, feedback: str, user_id: int, message_id: int):
        with self.Session() as session:
            new_feedback = Likes(
                user_id=user_id,
                message_id=message_id,
                feedback=feedback,
                is_correct=True,
            )
            session.add(new_feedback)
            session.commit()

    def count_user_messages(self, user_id: int, model: str, interval: int):
        with self.Session() as session:
            current_ts = self.get_current_ts()
            count = (
                session.query(func.count(Messages.id))
                .filter(
                    Messages.reply_user_id == user_id,
                    Messages.role == "assistant",
                    Messages.model == model,
                    Messages.timestamp.isnot(None),
                    Messages.timestamp > (current_ts - interval),
                )
                .scalar()
            )
            return count

    def get_all_conv_ids(self):
        with self.Session() as session:
            conversations = session.query(Conversations).all()
            return [conv.conv_id for conv in conversations]

    def _serialize_content(self, content):
        if isinstance(content, str):
            return content
        return json.dumps(content)

    def _parse_content(self, content):
        try:
            parsed_content = json.loads(content)
            if not isinstance(parsed_content, list):
                return content
            for m in parsed_content:
                if not isinstance(m, dict):
                    return content
            return parsed_content
        except json.JSONDecodeError:
            return content
