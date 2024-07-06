import os
from typing import Dict

from transformers import AutoTokenizer  # type: ignore

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Tokenizers:
    tokenizers: Dict[str, AutoTokenizer] = dict()

    @classmethod
    def get(cls, model_name: str) -> AutoTokenizer:
        if model_name not in cls.tokenizers:
            cls.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
        return cls.tokenizers[model_name]
