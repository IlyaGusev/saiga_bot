from abc import ABC, abstractmethod
from typing import Any, Union, List, Dict

from src.registrable import Registrable


class Tool(ABC, Registrable):
    @abstractmethod
    async def __call__(self, *args: Any, **kwargs: Any) -> Union[str, List[Dict[str, Any]]]:
        raise NotImplementedError()

    @abstractmethod
    def get_specification(self) -> Dict[str, Any]:
        raise NotImplementedError()
