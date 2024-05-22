import os
import json

import fire
from tinydb import TinyDB
from tqdm import tqdm
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database import Base, Conversations, Messages, SystemPrompts, Models, Likes
from database import Database, Conversations, Messages, SystemPrompts, Models, Likes


def migrate_tinydb_to_sqlite(tinydb, sqlite_db):
    engine = create_engine(f'sqlite:///{sqlite_db}')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Migrate conversations
    for conv in tqdm(tinydb.table("current_conversations").all()):
        session.add(Conversations(
            user_id=conv['user_id'],
            conv_id=conv['conv_id'],
            timestamp=conv['timestamp']
        ))
    session.commit()

    # Migrate messages
    for msg in tqdm(tinydb.table("messages").all()):
        if msg['content'] is None:
            continue
        if not isinstance(msg['conv_id'], str):
            continue
        if isinstance(msg['content'], str):
            content = msg['content']
        elif isinstance(msg['content'], list):
            content = json.dumps(msg["content"], ensure_ascii=False)
        else:
            content = str(msg['content'])
        session.add(Messages(
            role=msg['role'],
            content=content,
            conv_id=msg['conv_id'],
            timestamp=msg['timestamp'],
            message_id=msg.get('message_id'),
            model=msg.get('model'),
            system_prompt=msg.get('system_prompt')
        ))
    session.commit()

    # Migrate system prompts
    for prompt in tinydb.table("system_prompts").all():
        session.add(SystemPrompts(
            user_id=prompt['user_id'],
            prompt=prompt['prompt']
        ))
    session.commit()

    # Migrate models
    for model in tinydb.table("models").all():
        session.add(Models(
            user_id=model['user_id'],
            model=model['model']
        ))
    session.commit()

    # Migrate likes
    for like in tinydb.table("likes").all():
        session.add(Likes(
            user_id=like['user_id'],
            message_id=like['message_id'],
            feedback=like['feedback'],
            is_correct=like.get('is_correct', False)
        ))
    session.commit()


def migrate(input_path, output_path):
    tinydb = TinyDB(input_path, ensure_ascii=False)
    migrate_tinydb_to_sqlite(tinydb, output_path)


if __name__ == "__main__":
    fire.Fire(migrate)
