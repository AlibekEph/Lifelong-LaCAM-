# strategies/ordering/base.py
from __future__ import annotations
from typing import Protocol, List

from core.configuration import Configuration
from core.hl_node import HLNode
from core.graph.base import GraphBase


class AgentOrdering(Protocol):
    """
    Интерфейс стратегии сортировки агентов по приоритету.

    Где это используется:
        - HLNode.order — порядок агентов
        - PIBT → pri[a] = порядок в списке
        - LaCAM расширяет LL-constraints по этому порядку
    """

    def init_order(
        self,
        starts: List[int],
        goals: List[int],
        graph: GraphBase,
    ) -> List[int]:
        """
        Начальное упорядочивание агентов на шаге root HL-узла.
        Должно вернуть список всех агентов в порядке приоритета.

        Обычно на старте приоритет задаётся расстоянием до цели
        или просто range(n_agents).
        """
        ...

    def reorder(
        self,
        config: Configuration,
        prev_order: List[int],
        parent: HLNode,
        graph: GraphBase,
        goals: List[int],
    ) -> List[int]:
        """
        Переупорядочивание агентов при переходе к новому HL-узлу.

        В LaCAM обычно оставляют порядок прежним (prev_order),
        но можно адаптировать на основе новой конфигурации.

        Возвращает новый список агентов в порядке приоритета.
        """
        ...
