# Based on https://github.com/allenai/allennlp/blob/master/allennlp/common/registrable.py
from typing import Callable, Dict, Type, TypeVar, ClassVar, Generic, Any
from collections import defaultdict


class Registrable:
    _registry: ClassVar[Dict[Type["Registrable"], Dict[str, Type["Registrable"]]]] = defaultdict(dict)

    @classmethod
    def register(cls, name: str) -> Callable[[Type[Any]], Type[Any]]:
        registry = Registrable._registry[cls]

        def add_subclass_to_registry(subclass: Type[Any]) -> Type[Any]:
            if name in registry:
                message = (
                    f"Cannot register {name} as {cls.__name__}; " f"name already in use for {registry[name].__name__}"
                )
                raise RuntimeError(message)
            registry[name] = subclass
            return subclass

        return add_subclass_to_registry

    @classmethod
    def by_name(cls, name: str) -> Type[Any]:
        result = Registrable._registry[cls].get(name)
        if result is None:
            raise RuntimeError(f"{name} is not a registered name for {cls.__name__}.")
        return result
