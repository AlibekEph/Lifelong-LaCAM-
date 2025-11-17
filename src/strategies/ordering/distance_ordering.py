# strategies/ordering/distance_ordering.py
from __future__ import annotations
from typing import List

from core.configuration import Configuration
from core.hl_node import HLNode
from core.graph.base import GraphBase

from .base import AgentOrdering


class DistanceOrdering(AgentOrdering):
    """
    Агент с большим расстоянием до своей цели получает БОЛЬШИЙ приоритет.
    Это стандартная и эффективная эвристика для PIBT/LaCAM.
    """

    def init_order(
        self,
        starts: List[int],
        goals: List[int],
        graph: GraphBase,
    ) -> List[int]:

        n = len(starts)

        # считаем расстояние: dist(start[i] → goal[i])
        dist_list = []
        for i in range(n):
            d = graph.dist(starts[i], goals[i])
            # если цели недостижимы (graph.dist == -1), ставим лучший приоритет
            if d < 0:
                d = 10**9
            dist_list.append((d, i))

        # сортируем: больше дистанция → выше приоритет → раньше в списке
        dist_list.sort(key=lambda x: -x[0])

        return [i for _, i in dist_list]

    def reorder(
        self,
        config: Configuration,
        prev_order: List[int],
        parent: HLNode,
        graph: GraphBase,
        goals: List[int],
    ) -> List[int]:
        """
        Стандартное поведение LaCAM: НЕ переназначать порядок,
        иначе это ломает свойства LL-constraints.

        Но при желании можно делать динамическое reorder:
            - dist(config[i] → goals[i])
            - last_conflicts
            - набегание очереди агента
        """

        # Классический LaCAM просто оставляет порядок прежним
        return prev_order
