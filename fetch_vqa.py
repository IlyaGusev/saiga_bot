import json

import fire
from database import Database


def main(db_path, output_path):
    db = Database(db_path)
    conversations = db.get_all_conv_ids()
    records = []
    for conv_id in conversations:
        messages = db.fetch_conversation(conv_id)
        has_image = False
        for m in messages:
            if m["content"] and not isinstance(m["content"], str):
                has_image = True
        if not has_image:
            continue
        records.append({
            "messages": messages
        })

    with open(output_path, "w") as w:
        for record in records:
            w.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    fire.Fire(main)
