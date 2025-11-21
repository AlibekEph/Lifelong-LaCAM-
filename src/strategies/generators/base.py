from __future__ import annotations
from typing import Protocol, Optional, Callable

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
        task_callback: Optional[Callable[[int, int, int], int]] = None,
        window: int = 1,
        allow_goal_callback: bool = False,
        agent_done: Optional[list[bool]] = None,
    ) -> Optional[Configuration]:
        """
        Попытаться построить новую Configuration, учитывая:
            - текущую конфигурацию hl_node.config
            - LL-constraint (кто → куда)
            - топологию графа
            - порядок агентов (hl_node.order)
            - коллизии между агентами
            - при необходимости вызвать task_callback, если агентов нужно
              отправить на новую цель (используется при оконном планировании)
              Вызов происходит только если allow_goal_callback=True, обычно
              когда генератор работает в окне >1 и нужно сразу знать
              следующую цель для эвристик.

        Возвращает:
            Configuration — если удалось построить допустимую конфигурацию
            None — если конфигурация недопустима или конфликтует
        """
        ...
