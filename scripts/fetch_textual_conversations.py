import json

import fire
from tqdm import tqdm

from src.database import Database


def merge_messages(messages):
    new_messages = []
    prev_role = None
    for m in messages:
        if m["content"] is None:
            continue
        if m["role"] == prev_role:
            new_messages[-1]["content"] += "\n\n" + m["content"]
            continue
        prev_role = m["role"]
        new_messages.append(m)
    return new_messages


def main(db_path: str, output_path: str, min_timestamp: int = None, fetch_chats: bool = False):
    db = Database(db_path)
    conversations = set(db.get_all_conv_ids(min_timestamp=min_timestamp))
    records = []
    first_messages = set()
    for conv_id in tqdm(conversations):
        messages = db.fetch_conversation(conv_id)

        timestamps = {m.get("timestamp", None) for m in messages}
        if not timestamps:
            continue
        if min_timestamp and min_timestamp > min(timestamps):
            continue

        user_ids = {m.get("user_id", None) for m in messages}
        if None in user_ids:
            user_ids.remove(None)
        if not fetch_chats and len(user_ids) > 1:
            continue
        if fetch_chats and len(user_ids) <= 1:
            continue

        models = {m.get("model") for m in messages}
        if None in models:
            models.remove(None)
        if len(models) != 1:
            continue
        model = list(models)[0]

        system_prompts = {m.get("system_prompt") for m in messages}

        if None in system_prompts:
            system_prompts.remove(None)
        if len(system_prompts) != 1:
            continue
        system_prompt = list(system_prompts)[0]

        if not all(isinstance(m["content"], str) for m in messages):
            continue
        if not fetch_chats:
            messages = [{"role": m["role"], "content": m["content"]} for m in messages]
            messages = merge_messages(messages)
        else:
            for message in messages:
                if message["role"] != "user":
                    continue
                user_name = message["user_name"] if message["user_name"] else "Неизвестный"
                message["content"] = "{}: {}".format(user_name, message["content"])
            messages = [{"role": m["role"], "content": m["content"]} for m in messages]
            messages = merge_messages(messages)

        if messages[0]["role"] != "user":
            continue

        while messages[-1]["role"] != "assistant":
            messages = messages[:-1]

        first_message = messages[0]["content"]
        if first_message in first_messages and len(messages) <= 2:
            continue
        first_messages.add(first_message)

        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        model_name = model.replace('-', '_')
        if model_name not in ("gpt_4o", "claude_3_5_sonnet", "o1_mini", "o1_preview", "claude_3_5_sonnet_20241022", "claude_opus", "deepseek_v3", "grok_2"):
            continue
        records.append({
            "messages": messages,
            "source": f"saiga_bot_28_02_{model_name}",
            "conv_id": conv_id
        })

    with open(output_path, "w") as w:
        for record in records:
            w.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    fire.Fire(main)
