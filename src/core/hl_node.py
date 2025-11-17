from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from collections import deque

from .configuration import Configuration
from .constraint import Constraint


@dataclass
class HLNode:
    """
    High-Level Node (HL-node) в алгоритме LaCAM.

    Содержит:
        config           : Configuration
            Конфигурация всех агентов в этой точке поиска.

        constraint_tree  : deque[Constraint]
            Очередь LL-узлов (узлов дерева ограничений),
            которые LaCAM будет по одному извлекать (BFS).

        order            : list[int]
            Порядок агентов для генератора конфигураций (PIBT/Greedy).

        parent           : HLNode | None
            Родительский HL-узел (для восстановления пути).
    """

    config: Configuration
    constraint_tree: deque[Constraint]
    order: list[int]
    parent: Optional["HLNode"]

    def is_goal(self, goal_config: Configuration) -> bool:
        """Проверка: достигли ли мы конфигурации целей."""
        return self.config == goal_config

    def reconstruct_path(self) -> list[Configuration]:
        """
        Восстановить полный путь конфигураций от корня до этого HL-узла.
        Используется, когда LaCAM находит цель.
        """
        path = []
        node = self
        while node is not None:
            path.append(node.config)
            node = node.parent
        return list(reversed(path))

    def depth(self) -> int:
        """Глубина HL-узла (для статистики/отладки)."""
        d = 0
        node = self.parent
        while node is not None:
            d += 1
            node = node.parent
        return d

    # Красивое представление HL-узла
    def __repr__(self):
        return f"HLNode(config={self.config}, constraints={len(self.constraint_tree)})"
