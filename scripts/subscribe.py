import fire

from src.database import Database


def subscribe(db_path: str, user_name: str = None, user_id: int = None, duration: int = 86400):
    db = Database(db_path)
    if user_name:
        user_id = db.get_user_id(user_name)
    db.subscribe_user(user_id, duration)


if __name__ == "__main__":
    fire.Fire(subscribe)
