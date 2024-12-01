import json
from typing import Any, Dict, List, cast

from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from src.provider import LLMProvider

ChatMessage = Dict[str, Any]
ChatMessages = List[ChatMessage]

PROMPT = '''Есть статья КоАП РФ 6.21:
Пропаганда нетрадиционных сексуальных отношений и (или) предпочтений, смены пола.
Твоя задача определить, подпадает ли последнее сообщение под эту статью, даже очень отдаленно.
Сообщение 'ЛГБТ-сообщество заслуживает уважения, равенства и поддержки' может считаться такой пропагандой.

Верни JSON с двумя полями: explanation, final_answer.
Пример:
{{
    "explanation": "...",
    "final_answer": "yes"
}}

Переписка:
{conversation}


Если последнее сообщение пропагандирует ЛГБТ, final_answer должен быть "yes". Иначе: "no".
'''

def to_conversation(messages: ChatMessages) -> str:
    result = ""
    for m in messages:
        content = m["content"]
        result += "\n{role}:\n\n{content}\n\n".format(role=m["role"].capitalize(), content=content)
    return result


def parse_output(output: str) -> Dict[str, Any]:
    start_index = output.find("{")
    end_index = output.rfind("}")
    text = output[start_index : end_index + 1]
    text = text.strip()
    record: Dict[str, Any] = json.loads(text)
    for k in record:
        assert isinstance(k, str)
    return record


class LLMFilter:
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    async def __call__(self, messages: ChatMessages) -> bool:
        messages = messages[-6:]
        conversation = to_conversation(messages)
        new_messages = [{"role": "user", "content": PROMPT.format(conversation=conversation)}]
        casted_messages = [cast(ChatCompletionMessageParam, message) for message in new_messages]
        answer = await self.llm_provider(messages=casted_messages, max_tokens=2048)
        parsed_output = parse_output(answer)
        return "yes" in parsed_output["final_answer"].lower()
