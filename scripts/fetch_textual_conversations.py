import json

import fire

from src.database import Database


def merge_messages(messages):
    new_messages = []
    prev_role = None
    for m in messages:
        if m["content"] is None:
            continue
        if m["role"] == prev_role:
            new_messages[-1]["content"] += "\n" + m["content"]
            continue
        prev_role = m["role"]
        new_messages.append(m)
    return new_messages


def main(db_path: str, output_path: str, min_timestamp: int = None):
    db = Database(db_path)
    conversations = set(db.get_all_conv_ids())
    records = []
    first_messages = set()
    for conv_id in conversations:
        messages = db.fetch_conversation(conv_id, include_meta=True)
        models = {m.get("model") for m in messages}
        system_prompts = {m.get("system_prompt") for m in messages}
        timestamps = {m.get("timestamp", None) for m in messages}
        if min_timestamp:
            if not timestamps:
                continue
            if min_timestamp > min(timestamps):
                continue
        if None in models:
            models.remove(None)
        if None in system_prompts:
            system_prompts.remove(None)
        if len(models) != 1:
            continue
        if len(system_prompts) != 1:
            continue
        model = list(models)[0]
        system_prompt = list(system_prompts)[0]
        messages = [{"role": m["role"], "content": m["content"]} for m in messages]
        if not all(isinstance(m["content"], str) for m in messages):
            continue
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
        records.append({
            "messages": messages,
            "model": model,
            "source": "saiga_bot_gpt_4o",
            "conv_id": conv_id
        })

    with open(output_path, "w") as w:
        for record in records:
            w.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    fire.Fire(main)
