from dataclasses import dataclass
from typing import Callable, List, Any


@dataclass
class Message:
    """Minimal message class used by MAS memory components."""

    role: str
    content: str


# LLMCallable: arbitrary kwargs allowed, returns a string.
LLMCallable = Callable[..., str]

