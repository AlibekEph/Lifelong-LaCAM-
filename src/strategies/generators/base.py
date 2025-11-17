from __future__ import annotations
from typing import Protocol, Optional

from core.configuration import Configuration
from core.constraint import Constraint
from core.hl_node import HLNode
from core.graph.base import GraphBase


class ConfigGenerator(Protocol):
    """
    Интерфейс генератора конфигураций для LaCAM.

    Генератор отвечает за:
        - построение новой Configuration на основе текущего HL-узла
        - учёт всех ограничений LL-узла (Constraint)
        - проверку коллизий (vertex/edge conflicts)
        - учет порядка агентов (hl_node.order)
        - выполнение одного шага, как в PIBT / Greedy

    Важно:
        generate() делает РОВНО ОДИН шаг для всех агентов.
        Никакого многоклеточного планирования здесь нет.
    """

    def generate(
        self,
        hl_node: HLNode,
        constraint: Constraint,
        graph: GraphBase,
    ) -> Optional[Configuration]:
        """
        Попытаться построить новую Configuration, учитывая:
            - текущую конфигурацию hl_node.config
            - LL-constraint (кто → куда)
            - топологию графа
            - порядок агентов (hl_node.order)
            - коллизии между агентами

        Возвращает:
            Configuration — если удалось построить допустимую конфигурацию
            None — если конфигурация недопустима или конфликтует
        """
        ...
