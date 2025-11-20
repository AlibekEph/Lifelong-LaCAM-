"""
Lifelong LaCAM со встроенной логикой обновления целей.

В отличие от наивной реализации, здесь логика Lifelong встроена
прямо в алгоритм LaCAM:

- Когда создаём новую конфигурацию, проверяем кто достиг цели
- СРАЗУ обновляем цель этого агента через callback
- Продолжаем поиск с обновлёнными целями
- НЕ ДЕЛАЕМ replanning снаружи - всё внутри одного run()
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Iterable, Callable, List
from collections import deque
import time

from .configuration import Configuration
from .constraint import Constraint
from .hl_node import HLNode
from .graph.base import GraphBase

from strategies.generators.base import ConfigGenerator
from strategies.ordering.base import AgentOrdering
from strategies.open_policy.base import OpenPolicy


# Callback для назначения новой цели
# (agent_id, current_pos, old_goal) -> new_goal
TaskCallback = Callable[[int, int, int], int]


@dataclass
class LifelongLaCAMIntegrated:
    """
    Lifelong LaCAM со встроенной логикой обновления целей.
    
    Ключевое отличие от базового LaCAM:
    - При создании новой конфигурации проверяем, кто достиг цели
    - Сразу обновляем цель через callback
    - Продолжаем поиск с новыми целями
    - Один непрерывный run(), без внешнего replanning
    
    Параметры:
        graph: GraphBase
        starts: list[int] - стартовые позиции
        initial_goals: list[int] - начальные цели
        generator: ConfigGenerator
        ordering: AgentOrdering
        open_policy: OpenPolicy
        task_callback: Callable - функция назначения новых целей
        reinsert: bool
        max_tasks_per_agent: int - максимум задач на агента (условие остановки)
    """
    
    graph: GraphBase
    starts: list[int]
    initial_goals: list[int]
    generator: ConfigGenerator
    ordering: AgentOrdering
    open_policy: OpenPolicy
    task_callback: TaskCallback
    reinsert: bool = False
    max_tasks_per_agent: int = 10  # условие остановки
    occupancy_penalty: float = 5.0
    backtrack_penalty: float = 1.0
    stay_bonus: float = 0.25
    enable_clustering: bool = True
    cluster_radius: int = 1
    
    def __post_init__(self):
        assert len(self.starts) == len(self.initial_goals)
        
        # Текущие цели (будут динамически обновляться)
        self.goals = list(self.initial_goals)
        self.start_config = Configuration(tuple(self.starts))
        self.goal_config = Configuration(tuple(self.goals))  # будет обновляться!
        self.num_agents = len(self.starts)
        
        # Статистика выполненных задач
        self.completed_tasks_count = [0] * self.num_agents
        self.completed_tasks_history: List[List[int]] = [[] for _ in range(self.num_agents)]
        # Флаг: агент уже получил награду за достижение текущей цели
        self._goal_completion_ack = [False] * self.num_agents
        # Агент достиг лимита задач и больше не участвует в выдаче целей
        self._agent_done = [False] * self.num_agents
        # Метрики HL-поиска
        self._hl_metrics = {
            'hl_nodes_created': 1,  # root HL-node
            'hl_revisited_nodes': 0,
            'll_expansions': 0,
            'll_nodes_created': 1,  # root constraint
            'max_constraint_queue': 1,
            'generator_failures': 0,
            'generator_successes': 0,
            'iterations': 0,
            'runtime_seconds': 0.0,
            'cluster_attempts': 0,
            'cluster_successes': 0,
            'cluster_fallbacks': 0,
            'max_clusters': 1,
            'max_cluster_size': self.num_agents,
        }
        
        # Explored таблица
        self._explored: Dict[tuple[Configuration, tuple[int, ...]], HLNode] = {}
        
        # Инициализация root node
        root_constraint = Constraint(parent=None, who=None, where=None, depth=0)
        constraint_tree = deque([root_constraint])
        
        init_order = self.ordering.init_order(
            starts=self.starts,
            goals=self.goals,
            graph=self.graph,
        )
        init_order = self._demote_done_agents(init_order)
        
        root_node = HLNode(
            config=self.start_config,
            constraint_tree=constraint_tree,
            order=list(init_order),
            parent=None,
        )
        
        self._explored[self._state_key(self.start_config, self.goals)] = root_node
        self.open_policy.push(root_node)
        
        # Статистика
        self.goal_updates_count = 0
        self.total_iterations = 0
    
    def run(self, max_iterations: Optional[int] = None, verbose: bool = False) -> Optional[list[Configuration]]:
        """
        Запуск Lifelong LaCAM со встроенной логикой обновления целей.
        
        Алгоритм продолжает работу пока:
        - Не достигнут max_iterations
        - Не выполнено max_tasks_per_agent задач каждым агентом
        
        Returns:
            Полный путь всех конфигураций (если нашли решение)
            None если не смогли продолжить
        """
        start_time = time.time()
        try:
            iterations = 0
            
            while not self.open_policy.empty():
                if max_iterations is not None and iterations >= max_iterations:
                    break
                
                iterations += 1
                self.total_iterations = iterations
                self._hl_metrics['iterations'] = iterations
                
                hl_node = self.open_policy.peek()
                
                # Проверка: все ли агенты выполнили достаточно задач?
                if self._check_stopping_condition():
                    return hl_node.reconstruct_path()
                
                # Стандартная логика LaCAM
                if not hl_node.constraint_tree:
                    self.open_policy.pop()
                    continue
                
                ll_node = hl_node.constraint_tree.popleft()
                self._hl_metrics['ll_expansions'] += 1

                # Перед генерацией шага прокидываем текущие цели в генератор, если поддерживается
                if hasattr(self.generator, "set_current_goals"):
                    try:
                        self.generator.set_current_goals(self.goals)
                    except Exception:
                        pass

                pos_to_agent = {pos: idx for idx, pos in enumerate(hl_node.config.pos)}

                # Расширяем constraint tree
                if ll_node.depth < self.num_agents:
                    agent_idx = hl_node.order[ll_node.depth]
                    current_pos = hl_node.config[agent_idx]
                    
                    next_vertices: list[int] = list(self.graph.neighbors(current_pos))
                    if not self.graph.is_blocked(current_pos) and current_pos not in next_vertices:
                        next_vertices.append(current_pos)

                    next_vertices.sort(
                        key=lambda v: self._constraint_score(
                            hl_node=hl_node,
                            agent_idx=agent_idx,
                            current_pos=current_pos,
                            candidate=v,
                            pos_to_agent=pos_to_agent,
                        )
                    )
                    
                    for u in next_vertices:
                        child = Constraint(
                            parent=ll_node,
                            who=agent_idx,
                            where=u,
                            depth=ll_node.depth + 1,
                        )
                        hl_node.constraint_tree.append(child)
                        self._hl_metrics['ll_nodes_created'] += 1
                        self._hl_metrics['max_constraint_queue'] = max(
                            self._hl_metrics['max_constraint_queue'],
                            len(hl_node.constraint_tree),
                        )

                # Генерируем новую конфигурацию (сначала пробуем кластеризацию)
                new_config = None
                if self.enable_clustering:
                    new_config = self._generate_clustered_step(
                        hl_node=hl_node,
                        ll_node=ll_node,
                        graph=self.graph,
                    )

                if new_config is None:
                    new_config = self.generator.generate(
                        hl_node=hl_node,
                        constraint=ll_node,
                        graph=self.graph,
                    )
                
                if new_config is None:
                    self._hl_metrics['generator_failures'] += 1
                    continue
                
                self._hl_metrics['generator_successes'] += 1
                
                # ⭐ КЛЮЧЕВАЯ ЛОГИКА LIFELONG ⭐
                # Проверяем, кто достиг цели в новой конфигурации
                goals_updated = self._check_and_update_goals(new_config, verbose)
                
                # Если обновили цели, обновляем goal_config
                if goals_updated:
                    self.goal_config = Configuration(tuple(self.goals))
                    
                    if verbose:
                        print(f"  Итерация {iterations}: Обновлены цели. "
                              f"Текущие цели: {self.goals}")
                
                # Проверка: может быть новая конфигурация уже удовлетворяет всем условиям?
                if self._check_stopping_condition():
                    if verbose:
                        print(f"\n✓ Достигнуто условие остановки!")
                    # Создаём путь и возвращаем
                    path = hl_node.reconstruct_path()
                    path.append(new_config)
                    return path
                
                # Стандартная обработка: проверка explored
                existing_node = self._explored.get(self._state_key(new_config, self.goals))
                if existing_node is not None:
                    hl_node.neighbors.add(existing_node)
                    existing_node.neighbors.add(hl_node)
                    self._hl_metrics['hl_revisited_nodes'] += 1
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
                self._hl_metrics['ll_nodes_created'] += 1
                self._hl_metrics['max_constraint_queue'] = max(
                    self._hl_metrics['max_constraint_queue'],
                    len(child_tree),
                )
                
                # ВАЖНО: используем текущие (возможно обновлённые) цели
                new_order = self.ordering.reorder(
                    config=new_config,
                    prev_order=hl_node.order,
                    parent=hl_node,
                    graph=self.graph,
                    goals=self.goals,  
                )
                new_order = self._demote_done_agents(new_order)
                
                edge_cost = self.get_edge_cost(hl_node.config, new_config)

                child_node = HLNode(
                    config=new_config,
                    constraint_tree=child_tree,
                    order=new_order,
                    parent=hl_node,
                    cost_from_parent=edge_cost,
                )
                hl_node.neighbors.add(child_node)
                child_node.neighbors.add(hl_node)
                self._hl_metrics['hl_nodes_created'] += 1
                self._explored[self._state_key(new_config, self.goals)] = child_node
                self.open_policy.push(child_node)
            
            # Не нашли решение
            if verbose:
                print(f"\n⚠️  Open пуст после {iterations} итераций")
                print(f"   Задач выполнено: {self.completed_tasks_count}")
            
            return None
        finally:
            self._hl_metrics['runtime_seconds'] = time.time() - start_time
    
    def _state_key(self, config: Configuration, goals_snapshot: Optional[List[int]] = None) -> tuple[Configuration, tuple[int, ...]]:
        """
        Ключ для explored: учитываем конфигурацию и актуальные цели,
        чтобы после смены goal'ов переобход не блокировался старыми узлами.
        """
        goals_tuple = tuple(goals_snapshot if goals_snapshot is not None else self.goals)
        return (config, goals_tuple)

    def _demote_done_agents(self, order: list[int]) -> list[int]:
        """
        Переносит агентов, у которых закончились задания, в конец порядка,
        чтобы их можно было вытеснять приоритетом PIBT и не держать коридоры.
        """
        active = [aid for aid in order if not self._agent_done[aid]]
        finished = [aid for aid in order if self._agent_done[aid]]
        return active + finished

    # ------------------------------------------------------------
    # Кластеры и вспомогательные функции LL
    # ------------------------------------------------------------
    def _collect_positive_constraints(self, constraint: Constraint) -> tuple[dict[int, int], list[int], list[Constraint]]:
        """
        Собрать все positive constraints (who -> where) из цепочки LL-узлов от корня до constraint.
        Возвращает:
            forced_moves: who -> where
            forced_order: порядок добавления ограничений
            chain: список Constraint от корня до constraint (включительно)
        """
        chain = constraint.path()
        forced: dict[int, int] = {}
        order: list[int] = []
        for node in chain:
            if node.who is None:
                continue
            forced[node.who] = node.where  # type: ignore[assignment]
            order.append(node.who)
        return forced, order, chain

    def _clone_constraint_chain(self, base_chain: list[Constraint], extra_forced: list[tuple[int, int]]) -> Constraint:
        """
        Склонировать цепочку Constraint (root -> leaf) и добавить дополнительные forced-узлы.
        Возвращает leaf новой цепочки.
        """
        parent: Optional[Constraint] = None
        depth = 0
        # корень
        parent = Constraint(parent=None, who=None, where=None, depth=depth)
        # основная цепочка (пропускаем старый корень)
        for node in base_chain[1:]:
            depth += 1
            parent = Constraint(parent=parent, who=node.who, where=node.where, depth=depth)
        # дополнительные ограничения
        for who, where in extra_forced:
            depth += 1
            parent = Constraint(parent=parent, who=who, where=where, depth=depth)
        return parent

    def _cluster_reachable(self, aid: int, config: Configuration, forced_moves: dict[int, int], graph: GraphBase) -> set[int]:
        """
        Множество позиций, способных быть занятых агентом на t+1 (радиус cluster_radius).
        Используется для определения независимости кластеров.
        """
        start = config[aid]
        target = forced_moves.get(aid, start)
        reachable = {start, target}
        frontier = {start}
        for _ in range(self.cluster_radius):
            new_frontier = set()
            for v in frontier:
                for nb in graph.neighbors(v):
                    new_frontier.add(nb)
            reachable |= new_frontier
            frontier = new_frontier
        return reachable

    def _compute_clusters(self, config: Configuration, graph: GraphBase, forced_moves: dict[int, int]) -> list[list[int]]:
        """
        Разбиение агентов на независимые кластеры: пересечение reachable множеств -> ребро.
        """
        n = self.num_agents
        reach: list[set[int]] = [self._cluster_reachable(aid, config, forced_moves, graph) for aid in range(n)]
        clusters: list[list[int]] = []
        unassigned = set(range(n))
        while unassigned:
            seed = unassigned.pop()
            cluster = [seed]
            queue = [seed]
            while queue:
                i = queue.pop()
                intersect = []
                for j in list(unassigned):
                    if reach[i].intersection(reach[j]):
                        intersect.append(j)
                for j in intersect:
                    unassigned.remove(j)
                    cluster.append(j)
                    queue.append(j)
            clusters.append(cluster)
        return clusters

    def _validate_configuration(
        self,
        old_conf: Configuration,
        new_conf: Configuration,
        graph: GraphBase,
        forced_moves: dict[int, int],
    ) -> bool:
        """Глобальная проверка корректности шага (вершинные/ребровые коллизии, допустимость ходов, respect forced)."""
        # vertex collisions
        if len(set(new_conf.pos)) != len(new_conf.pos):
            return False
        # moves and forced
        for aid, (u, v) in enumerate(zip(old_conf.pos, new_conf.pos)):
            if graph.is_blocked(v):
                return False
            if u != v and v not in graph.neighbors(u):
                return False
            forced = forced_moves.get(aid)
            if forced is not None and forced != v:
                return False
        # edge swaps
        n = len(old_conf.pos)
        for i in range(n):
            for j in range(i + 1, n):
                if old_conf.pos[i] == new_conf.pos[j] and old_conf.pos[j] == new_conf.pos[i] \
                        and old_conf.pos[i] != old_conf.pos[j]:
                    return False
        return True

    def _generate_clustered_step(
        self,
        hl_node: HLNode,
        ll_node: Constraint,
        graph: GraphBase,
    ) -> Optional[Configuration]:
        """
        Попытка сделать LL-шага по независимым кластерам.
        Возвращает новую конфигурацию или None (тогда используем обычный генератор).
        """
        forced_moves, _, chain = self._collect_positive_constraints(ll_node)
        clusters = self._compute_clusters(hl_node.config, graph, forced_moves)
        if len(clusters) <= 1:
            return None

        self._hl_metrics['cluster_attempts'] += 1
        self._hl_metrics['max_clusters'] = max(self._hl_metrics['max_clusters'], len(clusters))
        for cl in clusters:
            self._hl_metrics['max_cluster_size'] = max(self._hl_metrics['max_cluster_size'], len(cl))

        cluster_configs: list[Configuration] = []
        for cluster in clusters:
            extra_forced: list[tuple[int, int]] = []
            for aid in range(self.num_agents):
                if aid in cluster:
                    continue
                if aid in forced_moves:
                    # уже зафиксирован ll-узлом
                    continue
                extra_forced.append((aid, hl_node.config[aid]))
            leaf = self._clone_constraint_chain(chain, extra_forced)
            conf = self.generator.generate(
                hl_node=hl_node,
                constraint=leaf,
                graph=graph,
            )
            if conf is None:
                # один кластер не смог сделать шаг — откатываемся к глобальному режиму
                self._hl_metrics['cluster_fallbacks'] += 1
                return None
            cluster_configs.append(conf)

        # Собираем итоговую конфигурацию
        new_pos: list[int] = [hl_node.config[aid] for aid in range(self.num_agents)]
        for cluster, conf in zip(clusters, cluster_configs):
            for aid in cluster:
                new_pos[aid] = conf[aid]

        combined = Configuration(tuple(new_pos))
        if not self._validate_configuration(hl_node.config, combined, graph, forced_moves):
            # предохранитель: если вдруг кластеры пересеклись
            self._hl_metrics['cluster_fallbacks'] += 1
            return None

        self._hl_metrics['cluster_successes'] += 1
        return combined

    def _constraint_score(
        self,
        hl_node: HLNode,
        agent_idx: int,
        current_pos: int,
        candidate: int,
        pos_to_agent: dict[int, int],
    ) -> float:
        goal = self.goals[agent_idx]
        dist = self.graph.dist(candidate, goal)
        if dist < 0:
            dist = float(self.graph.num_vertices())

        penalty = 0.0
        if candidate == current_pos:
            penalty -= self.stay_bonus
        else:
            occupant = pos_to_agent.get(candidate)
            if occupant is not None and occupant != agent_idx:
                penalty += self.occupancy_penalty

        if hl_node.parent is not None:
            prev_pos = hl_node.parent.config[agent_idx]
            if candidate == prev_pos and candidate != goal:
                penalty += self.backtrack_penalty

        return dist + penalty
    
    def _check_and_update_goals(self, config: Configuration, verbose: bool) -> bool:
        """
        Проверяет, кто достиг цели в данной конфигурации.
        Обновляет цели через callback для тех, кто достиг.
        
        Returns:
            True если были обновления, False иначе
        """
        goals_updated = False
        
        for agent_id in range(self.num_agents):
            current_pos = config[agent_id]
            current_goal = self.goals[agent_id]

            if self._agent_done[agent_id]:
                continue

            if current_pos != current_goal:
                # как только агент покидает цель, можно снова засчитывать достижение
                self._goal_completion_ack[agent_id] = False
                continue

            old_goal = current_goal
            new_goal = self.task_callback(agent_id, current_pos, old_goal)

            # учитываем достижение цели один раз, пока агент не покинет её
            if not self._goal_completion_ack[agent_id]:
                self.completed_tasks_count[agent_id] += 1
                self.completed_tasks_history[agent_id].append(old_goal)
                self._goal_completion_ack[agent_id] = True

                if verbose:
                    print(f"    Агент {agent_id}: завершил цель {old_goal} "
                          f"(всего задач: {self.completed_tasks_count[agent_id]})")

            if self.completed_tasks_count[agent_id] >= self.max_tasks_per_agent:
                self._agent_done[agent_id] = True
                # достигнут лимит задач — больше целей не выдаём
                continue

            if new_goal != old_goal:
                self.goals[agent_id] = new_goal
                self.goal_updates_count += 1
                self._goal_completion_ack[agent_id] = False
                goals_updated = True

                if verbose:
                    print(f"    Агент {agent_id}: новая цель {new_goal}")
        
        return goals_updated
    
    def _check_stopping_condition(self) -> bool:
        """
        Проверяет условие остановки:
        все ли агенты выполнили достаточно задач?
        """
        return all(count >= self.max_tasks_per_agent 
                   for count in self.completed_tasks_count)
    
    def get_statistics(self) -> dict:
        """Получить статистику работы."""
        generator_metrics = None
        if hasattr(self.generator, "get_metrics"):
            try:
                generator_metrics = self.generator.get_metrics()
            except Exception:
                generator_metrics = None
        return {
            'total_iterations': self.total_iterations,
            'goal_updates': self.goal_updates_count,
            'completed_tasks_per_agent': self.completed_tasks_count.copy(),
            'total_completed_tasks': sum(self.completed_tasks_count),
            'tasks_history': [tasks.copy() for tasks in self.completed_tasks_history],
            'hl_metrics': self._hl_metrics.copy(),
            'generator_metrics': generator_metrics,
            'runtime_seconds': self._hl_metrics.get('runtime_seconds'),
        }

    def get_edge_cost(self, config_from: Configuration, config_to: Configuration) -> int:
        """
        Стоимость перехода между конфигурациями: число агентов,
        которые не остаются на своей цели.
        """
        cost = 0
        for aid in range(self.num_agents):
            if not (self.goals[aid] == config_from[aid] == config_to[aid]):
                cost += 1
        return cost
