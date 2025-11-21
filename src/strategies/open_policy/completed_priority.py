from __future__ import annotations
from typing import List

from core.hl_node import HLNode
from .base import OpenPolicy


class CompletedPriorityOpen(OpenPolicy):
    """
    Open-список с приоритетом по суммарно выполненным задачам (completed_sum).
    При равном completed_sum работает как стек (LIFO).
    """

    def __init__(self):
        self._stack: List[HLNode] = []

    def _sort(self) -> None:
        # сортируем по возрастанию, pop/peek берут хвост (максимум completed_sum)
        self._stack.sort(key=lambda n: n.completed_sum)

    def push(self, node: HLNode) -> None:
        self._stack.append(node)

    def pop(self) -> HLNode:
        self._sort()
        return self._stack.pop()

    def peek(self) -> HLNode:
        self._sort()
        return self._stack[-1]

    def empty(self) -> bool:
        return len(self._stack) == 0
