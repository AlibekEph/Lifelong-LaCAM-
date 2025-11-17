
from __future__ import annotations
from typing import List

from core.hl_node import HLNode
from .base import OpenPolicy


class StackOpen(OpenPolicy):
    """
    Open-список как стек (LIFO).

    Это поведение соответствует классическому LaCAM.
    """

    def __init__(self):
        self._stack: List[HLNode] = []

    def push(self, node: HLNode) -> None:
        self._stack.append(node)

    def pop(self) -> HLNode:
        return self._stack.pop()

    def peek(self) -> HLNode:
        return self._stack[-1]

    def empty(self) -> bool:
        return len(self._stack) == 0
