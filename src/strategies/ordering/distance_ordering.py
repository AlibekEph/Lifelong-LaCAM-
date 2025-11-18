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
        Динамическое обновление порядка: дальше от цели → выше приоритет.
        При равенстве расстояний сохраняем прежний порядок, чтобы не
        ломать детерминизм и стабильность.
        """
        pos_prev = {ag: idx for idx, ag in enumerate(prev_order)}
        dist_list = []
        for aid in range(len(goals)):
            d = graph.dist(config[aid], goals[aid])
            if d < 0:
                d = 10**9
            dist_list.append((d, aid))

        dist_list.sort(key=lambda x: (-x[0], pos_prev.get(x[1], len(prev_order))))
        return [aid for _, aid in dist_list]
