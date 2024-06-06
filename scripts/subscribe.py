import fire

from src.database import Database


def subscribe(user_name: str, db_path: str, duration: int):
    db = Database(db_path)
    user_id = db.get_user_id(user_name)
    db.subscribe_user(user_id, duration)


if __name__ == "__main__":
    fire.Fire(subscribe)
