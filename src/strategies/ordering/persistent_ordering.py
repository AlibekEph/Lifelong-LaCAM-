# strategies/ordering/persistent_ordering.py
from __future__ import annotations
from typing import List, Optional

from core.configuration import Configuration
from core.hl_node import HLNode
from core.graph.base import GraphBase

from .base import AgentOrdering


class PersistentPriorityOrdering(AgentOrdering):
    """
    Агентам, которые дольше не достигают цели, назначается повышенный приоритет.

    Идея:
        - у каждого агента копится "возраст" приоритета (score)
        - на каждом HL-узле score увеличивается на boost, если агент не на цели
        - когда агент достигает цели, score сбрасывается в ноль
        - можно задать decay, чтобы score постепенно убывал (по умолчанию без затухания)

    Таким образом приоритет сохраняется между HL-узлами, пока агент не доберётся до цели.
    """

    def __init__(self, boost: float = 1.0, decay: float = 0.0) -> None:
        assert boost > 0.0, "boost должен быть > 0"
        assert decay >= 0.0, "decay должен быть >= 0"
        self.boost = boost
        self.decay = decay
        self._scores: Optional[List[float]] = None

    def init_order(
        self,
        starts: List[int],
        goals: List[int],
        graph: GraphBase,
    ) -> List[int]:
        n = len(starts)
        self._scores = [0.0 for _ in range(n)]

        dist_list = []
        for i in range(n):
            d = graph.dist(starts[i], goals[i])
            if d < 0:
                d = 10**9
            dist_list.append((d, i))

        dist_list.sort(key=lambda x: -x[0])
        return [idx for _, idx in dist_list]

    def reorder(
        self,
        config: Configuration,
        prev_order: List[int],
        parent: HLNode,
        graph: GraphBase,
        goals: List[int],
    ) -> List[int]:
        n = len(goals)
        if self._scores is None or len(self._scores) != n:
            self._scores = [0.0 for _ in range(n)]

        prev_pos = {agent: idx for idx, agent in enumerate(prev_order)}
        ordering_keys = []

        for aid in range(n):
            dist = graph.dist(config[aid], goals[aid])
            if dist < 0:
                dist = 10**9

            score = self._scores[aid]
            if config[aid] == goals[aid]:
                score = 0.0
            else:
                score = max(0.0, score - self.decay) + self.boost
            self._scores[aid] = score

            ordering_keys.append((
                (-score, -dist, prev_pos.get(aid, n + aid)),
                aid,
            ))

        ordering_keys.sort(key=lambda item: item[0])
        return [aid for _, aid in ordering_keys]
