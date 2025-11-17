
from __future__ import annotations
from typing import Protocol, Optional

from core.hl_node import HLNode


class OpenPolicy(Protocol):
    """
    Интерфейс управления Open-списком HL-узлов в LaCAM.

    OpenPolicy определяет:
      - как добавлять узлы (push)
      - как извлекать (pop / peek)
      - как проверять пустоту
      - возможно, как изменять положение узлов (reinsertion)

    Это чистый Strategy Pattern.
    """

    def push(self, node: HLNode) -> None:
        """Добавить HL-узел в структуру."""
        ...

    def pop(self) -> HLNode:
        """Удалить и вернуть следующий HL-узел."""
        ...

    def peek(self) -> HLNode:
        """Посмотреть на следующий HL-узел, НЕ удаляя."""
        ...

    def empty(self) -> bool:
        """True если Open-список пуст."""
        ...
