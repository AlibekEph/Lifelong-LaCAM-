from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Set, List
from collections import deque

from .configuration import Configuration
from .constraint import Constraint


@dataclass
class ClusterPlanSnapshot:
    """
    Снимок оконного планирования для кластера.

    Храним:
        agents               : список агентов кластера
        window               : длина окна (w), с которым планировался кластер
        constraint_chains    : список цепочек ограничений (root -> leaf) на каждый шаг окна
        path                 : глобальные конфигурации, полученные при планировании (len = window+1)
    """

    agents: List[int]
    window: int
    constraint_chains: List[List[Constraint]]
    path: List[Configuration]


@dataclass(eq=False)
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
        neighbors        : set[HLNode]
            Соседние HL-узлы (для переиспользования конфигураций).
    """

    config: Configuration
    constraint_tree: deque[Constraint]
    order: list[int]
    parent: Optional["HLNode"]
    cost_from_parent: int = 0
    neighbors: Set["HLNode"] = field(default_factory=set)
    completed_sum: int = 0
    # Доп. данные для кластеризации/оконного планирования (может быть None)
    cluster_plan: Optional[List[ClusterPlanSnapshot]] = None
    cluster_window_used: Optional[int] = None
    cluster_ll_states: Optional[list] = None  # заполняется LifelongLaCAMIntegrated
    cluster_cache: Optional[dict] = None  # cache успешных конфигураций по кластерам

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
