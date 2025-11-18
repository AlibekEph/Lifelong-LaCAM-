from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Iterable

from collections import deque

from core.configuration import Configuration
from core.constraint import Constraint
from core.hl_node import HLNode
from core.graph.base import GraphBase

from strategies.generators.base import ConfigGenerator
from strategies.ordering.base import AgentOrdering
from strategies.open_policy.base import OpenPolicy


@dataclass
class LaCAM:
    """
    Реализация алгоритма LaCAM (suboptimal, depth-first style).

    Параметры:
        graph         : GraphBase        — статический граф (grid / general)
        starts        : list[int]        — стартовые вершины агентов
        goals         : list[int]        — целевые вершины агентов
        generator     : ConfigGenerator  — генератор конфигураций (обычно PIBT)
        ordering      : AgentOrdering    — стратегия порядка агентов
        open_policy   : OpenPolicy       — структура Open (stack / priority queue)
        reinsert      : bool             — делать ли reinsertion уже известного HL-узла

    Главный метод:
        run() -> Optional[list[Configuration]]
    """

    graph: GraphBase
    starts: list[int]
    goals: list[int]
    generator: ConfigGenerator
    ordering: AgentOrdering
    open_policy: OpenPolicy
    reinsert: bool = True

    def __post_init__(self):
        assert len(self.starts) == len(self.goals), "starts и goals должны совпадать по длине"

        # конфигурации старта и цели
        self.start_config = Configuration(tuple(self.starts))
        self.goal_config = Configuration(tuple(self.goals))
        self.num_agents = len(self.starts)

        # таблица посещённых конфигураций (Explored)
        self._explored: Dict[Configuration, HLNode] = {}

        # инициализация корневого HL-узла
        root_constraint = Constraint(
            parent=None,
            who=None,
            where=None,
            depth=0,
        )
        constraint_tree = deque([root_constraint])

        init_order = self.ordering.init_order(
            starts=self.starts,
            goals=self.goals,
            graph=self.graph,
        )

        root_node = HLNode(
            config=self.start_config,
            constraint_tree=constraint_tree,
            order=list(init_order),
            parent=None,
        )

        self._explored[self.start_config] = root_node
        self.open_policy.push(root_node)

    # ------------------------------------------------------------
    # Публичный интерфейс
    # ------------------------------------------------------------
    def run(self, max_iterations: Optional[int] = None) -> Optional[list[Configuration]]:
        """
        Запустить LaCAM.

        Возвращает:
            список конфигураций от старта до цели, если решение найдено,
            иначе None.
        """
        while not self.open_policy.empty():
            # Ограничение теперь считается по числу уникальных HL-конфигураций,
            # а не по количеству Low-Level расширений (LL-узлов). Иначе поисковое
            # дерево LL может быстро исчерпать лимит, не попробовав достаточно HL-состояний.
            if max_iterations is not None and len(self._explored) >= max_iterations:
                break

            # Вариант depth-first: смотрим на верхушку Open, не убирая её,
            # пока не выработается всё LL-дерево (constraint_tree).
            hl_node = self.open_policy.peek()

            # Проверка цели
            if hl_node.config == self.goal_config:
                return hl_node.reconstruct_path()

            # Если у HL-узла больше нет LL-узлов — выкидываем его из Open
            if not hl_node.constraint_tree:
                self.open_policy.pop()  # тот же hl_node
                continue

            # Берём очередной LL-узел (constraint) из дерева (BFS)
            ll_node = hl_node.constraint_tree.popleft()

            # Перед генерацией шага прокидываем цели, если генератор поддерживает
            if hasattr(self.generator, "set_current_goals"):
                try:
                    self.generator.set_current_goals(self.goals)
                except Exception:
                    pass

            # Генерируем детей LL-узла (расширяем дерево ограничений),
            # если ещё не назначали ограничения для всех агентов.
            if ll_node.depth <= self.num_agents - 1:
                agent_idx = hl_node.order[ll_node.depth]  # depth 0 → первый агент
                current_pos = hl_node.config[agent_idx]

                # соседи + возможность остаться на месте
                next_vertices: Iterable[int] = list(self.graph.neighbors(current_pos))
                # stay
                if not self.graph.is_blocked(current_pos):
                    # избегаем дубликатов, если граф вдруг содержит петлю
                    if current_pos not in next_vertices:
                        next_vertices = list(next_vertices) + [current_pos]
                next_vertices.sort(key=lambda v: self.graph.dist(v, self.goals[agent_idx]))

                for u in next_vertices:
                    child = Constraint(
                        parent=ll_node,
                        who=agent_idx,
                        where=u,
                        depth=ll_node.depth + 1,
                    )
                    hl_node.constraint_tree.append(child)

            # Пытаемся сгенерировать новую конфигурацию для текущего LL-узла
            new_config = self.generator.generate(
                hl_node=hl_node,
                constraint=ll_node,
                graph=self.graph,
            )

            if new_config is None:
                # генератор не смог построить допустимую конфигурацию
                continue

            # Если конфигурация уже видели
            existing_node = self._explored.get(new_config)
            if existing_node is not None:
                # опциональный "reinsertion" — поднимаем узел в Open,
                # чтобы глубже исследовать его LL-дерево.
                if self.reinsert:
                    self.open_policy.push(existing_node)
                continue

            # Создаём новый HL-узел
            child_constraint_root = Constraint(
                parent=None,
                who=None,
                where=None,
                depth=0,
            )
            child_tree = deque([child_constraint_root])

            new_order = self.ordering.reorder(
                config=new_config,
                prev_order=hl_node.order,
                parent=hl_node,
                graph=self.graph,
                goals=self.goals,
            )

            child_node = HLNode(
                config=new_config,
                constraint_tree=child_tree,
                order=new_order,
                parent=hl_node,
            )

            self._explored[new_config] = child_node
            self.open_policy.push(child_node)

        # Если дошли сюда — решения нет (или вышли по max_iterations)
        return None
