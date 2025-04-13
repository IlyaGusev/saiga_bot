import base64
from typing import BinaryIO, Dict, Any, List, Union, Optional


IMAGE_PLACEHOLDER = "<image_placeholder>"

MessageContent = Union[str, List[Dict[str, Any]]]
ChatMessage = Dict[str, MessageContent]
ChatMessages = List[ChatMessage]


def is_image_content(content: MessageContent) -> bool:
    if isinstance(content, list):
        for item in content:
            if item["type"] == "image_url":
                return True
    return False


def replace_images(messages: ChatMessages) -> ChatMessages:
    for m in messages:
        content: MessageContent = m["content"]
        if is_image_content(content):
            texts: List[str] = []
            for item in content:
                assert isinstance(item, dict)
                if "text" in item:
                    texts.append(item["text"])
            m["content"] = IMAGE_PLACEHOLDER + "\n" + "\n".join(texts)
    return messages


def merge_messages(messages: ChatMessages) -> ChatMessages:
    new_messages: ChatMessages = []
    prev_role = None
    for m in messages:
        content: MessageContent = m["content"]
        assert isinstance(m["role"], str)
        role: str = m["role"]
        if role == prev_role:
            is_current_str = isinstance(content, str)
            is_current_list = isinstance(content, list)
            prev_content = new_messages[-1]["content"]
            is_prev_str = isinstance(prev_content, str)
            is_prev_list = isinstance(prev_content, list)
            if is_current_str and is_prev_str:
                assert isinstance(new_messages[-1]["content"], str)
                assert isinstance(content, str)
                new_messages[-1]["content"] += "\n\n" + content
                continue
            elif is_current_str and is_prev_list:
                assert isinstance(prev_content, list)
                assert isinstance(content, str)
                prev_content.append({"type": "text", "text": content})
                continue
            elif is_prev_str and is_current_list:
                assert isinstance(content, list)
                assert isinstance(prev_content, str)
                content.insert(0, {"type": "text", "text": prev_content})
                new_messages[-1]["content"] = content
                continue
            elif is_current_list and is_prev_list:
                assert isinstance(prev_content, list)
                assert isinstance(content, list)
                prev_content.extend(content)
                continue
        prev_role = role
        new_messages.append(m)
    return new_messages


def crop_content(content: MessageContent) -> str:
    if isinstance(content, str):
        return content.replace("\n", " ")[:40]
    return IMAGE_PLACEHOLDER


def build_image_content(caption: Optional[str], stream: BinaryIO) -> MessageContent:
    content: List[Dict[str, Any]] = []
    base64_image = base64.b64encode(stream.read()).decode("utf-8")
    assert base64_image
    if caption:
        content.append({"type": "text", "text": caption})
    content.append(
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
        }
    )
    return content


def format_chat(messages: ChatMessages) -> ChatMessages:
    for m in messages:
        content = m["content"]
        role = m["role"]
        if role == "user" and content is None:
            continue
        if role == "user" and isinstance(content, str) and m["user_name"]:
            m["content"] = "Из чата пишет {}: {}".format(m["user_name"], content)
    return messages
