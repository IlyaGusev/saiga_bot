import os
import json

import fire
from tinydb import TinyDB
from database import Database, Conversations, Messages, SystemPrompts, Models, Likes


def migrate_tinydb_to_sqlite(tinydb, sqlite_db):
    # Migrate conversations
    for conv in tinydb.table("current_conversations").all():
        sqlite_db.Session().add(Conversations(
            user_id=conv['user_id'],
            conv_id=conv['conv_id'],
            timestamp=conv['timestamp']
        ))
    sqlite_db.Session().commit()

    # Migrate messages
    for msg in tinydb.table("messages").all():
        sqlite_db.Session().add(Messages(
            role=msg['role'],
            content=msg['content'],
            conv_id=msg['conv_id'],
            timestamp=msg['timestamp'],
            message_id=msg.get('message_id'),
            model=msg.get('model'),
            system_prompt=msg.get('system_prompt')
        ))
    sqlite_db.Session().commit()

    # Migrate system prompts
    for prompt in tinydb.table("system_prompts").all():
        sqlite_db.Session().add(SystemPrompts(
            user_id=prompt['user_id'],
            prompt=prompt['prompt']
        ))
    sqlite_db.Session().commit()

    # Migrate models
    for model in tinydb.table("models").all():
        sqlite_db.Session().add(Models(
            user_id=model['user_id'],
            model=model['model']
        ))
    sqlite_db.Session().commit()

    # Migrate likes
    for like in tinydb.table("likes").all():
        sqlite_db.Session().add(Likes(
            user_id=like['user_id'],
            message_id=like['message_id'],
            feedback=like['feedback'],
            is_correct=like.get('is_correct', False)
        ))
    sqlite_db.Session().commit()


def migrate(input_path, output_path):
    tinydb = TinyDB(input_path, ensure_ascii=False)
    sqlite_db = Database(output_path)
    migrate_tinydb_to_sqlite(tinydb, sqlite_db)


if __name__ == "__main__":
    fire.Fire(migrate)
