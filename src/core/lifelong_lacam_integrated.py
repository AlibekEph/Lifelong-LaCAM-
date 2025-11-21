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
from typing import Optional, Dict, Iterable, Callable, List, Tuple
from collections import deque
import time

from .configuration import Configuration
from .constraint import Constraint
from .hl_node import HLNode, ClusterPlanSnapshot


@dataclass
class ClusterLLState:
    """Состояние LL-поиска для отдельного кластера."""
    cluster: list[int]
    order: list[int]
    constraint_tree: deque[tuple[Constraint, int]]  # (constraint, depth по cluster order)
    found_config: Optional[Configuration] = None
    failed: bool = False
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
        max_tasks_per_agent: int | None - максимум задач на агента (условие остановки), None = без лимита
    """
    
    graph: GraphBase
    starts: list[int]
    initial_goals: list[int]
    generator: ConfigGenerator
    ordering: AgentOrdering
    open_policy: OpenPolicy
    task_callback: TaskCallback
    reinsert: bool = False
    max_tasks_per_agent: Optional[int] = None  # None = без лимита
    occupancy_penalty: float = 5.0
    backtrack_penalty: float = 1.0
    stay_bonus: float = 0.25
    enable_clustering: bool = True
    cluster_radius: int = 1  # устарело, сохраняем для обратной совместимости
    cluster_window_w: int = 2  # максимальный размер окна для кластеризации/PIBT
    cluster_ll_limit: int = 100  # лимит на число узлов LL-поиска внутри кластера
    
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
            'cluster_window_attempts': 0,
            'last_cluster_window': None,
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
        root_node.completed_sum = sum(self.completed_tasks_count)
        
        self._explored[self._state_key(self.start_config, self.goals)] = root_node
        self.open_policy.push(root_node)
        
        # Статистика
        self.goal_updates_count = 0
        self.total_iterations = 0
        # буфер событий о смене целей, собранных генератором
        self._generator_goal_events: list[tuple[int, int, int]] = []
    
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
                self._generator_goal_events = []
                
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
                cluster_plan: Optional[list[ClusterPlanSnapshot]] = None
                cluster_window_used: Optional[int] = None
                if self.enable_clustering:
                    cluster_result = self._generate_clustered_step(
                        hl_node=hl_node,
                        ll_node=ll_node,
                        graph=self.graph,
                    )
                    clustered_config, cluster_plan, cluster_window_used, cluster_failed = cluster_result
                    if cluster_failed:
                        # полный фолбек: просто игнорируем кластеризацию и пробуем общий генератор
                        clustered_config = None
                        cluster_plan = None
                        cluster_window_used = None
                    if clustered_config is not None:
                        new_config = clustered_config

                if new_config is None:
                    new_config = self.generator.generate(
                        hl_node=hl_node,
                        constraint=ll_node,
                        graph=self.graph,
                        task_callback=self.task_callback,
                        window=self.cluster_window_w,
                        allow_goal_callback=True,
                        agent_done=self._agent_done,
                    )
                    self._collect_generator_goal_events()
                
                if new_config is None:
                    self._hl_metrics['generator_failures'] += 1
                    continue
                
                self._hl_metrics['generator_successes'] += 1
                
                # ⭐ КЛЮЧЕВАЯ ЛОГИКА LIFELONG ⭐
                # Проверяем, кто достиг цели в новой конфигурации
                goals_updated = self._apply_generator_goal_events(new_config, verbose)
                
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
                    cluster_plan=cluster_plan,
                    cluster_window_used=cluster_window_used,
                )
                child_node.completed_sum = sum(self.completed_tasks_count)
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

    def _cluster_reachable(self, aid: int, config: Configuration, forced_moves: dict[int, int], graph: GraphBase, window: int) -> set[int]:
        """
        Множество вершин, достижимых агентом за <= window шагов с учётом positive constraint первого шага.
        Используется для определения независимости кластеров.
        """
        start = config[aid]
        forced_first = forced_moves.get(aid)
        frontier = {forced_first} if forced_first is not None else {start}
        reachable = {start} | frontier
        steps = window
        while steps > 0:
            new_frontier: set[int] = set()
            for v in frontier:
                neigh = list(graph.neighbors(v))
                if not graph.is_blocked(v) and v not in neigh:
                    neigh.append(v)  # разрешаем stay
                for nb in neigh:
                    if graph.is_blocked(nb):
                        continue
                    new_frontier.add(nb)
            reachable |= new_frontier
            frontier = new_frontier
            steps -= 1
        return reachable

    def _compute_clusters(self, config: Configuration, graph: GraphBase, forced_moves: dict[int, int], window: int) -> list[list[int]]:
        """
        Разбиение агентов на независимые кластеры по пересечению reachable множеств на горизонте window.
        """
        n = self.num_agents
        reach: list[set[int]] = [self._cluster_reachable(aid, config, forced_moves, graph, window) for aid in range(n)]
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
    ) -> tuple[Optional[Configuration], Optional[list[ClusterPlanSnapshot]], Optional[int], bool]:
        """
        Пытаемся распараллелить LL-поиск по кластерам.
        Для каждого кластера строим собственное constraint_tree и сразу полностью
        его расширяем, пока не найдём конфигурацию или не исчерпаем дерево.
        Если хотя бы один кластер не нашёл конфигурацию — полный фолбек.
        """
        forced_moves, _, chain = self._collect_positive_constraints(ll_node)

        # Перебор окна, чтобы получить разбиение на кластеры
        window_used: Optional[int] = None
        clusters: list[list[int]] = []
        for window in range(self.cluster_window_w, 0, -1):
            clusters = self._compute_clusters(hl_node.config, graph, forced_moves, window)
            if len(clusters) <= 1:
                continue
            # защитное ограничение: если кластер слишком велик, пропускаем кластеризацию для этого окна
            if any(len(cl) > 4 for cl in clusters):
                continue
            window_used = window
            break

        if not clusters or window_used is None:
            return None, None, None, False

        self._hl_metrics['cluster_attempts'] += 1
        self._hl_metrics['cluster_window_attempts'] += 1
        self._hl_metrics['last_cluster_window'] = window_used
        self._hl_metrics['max_clusters'] = max(self._hl_metrics['max_clusters'], len(clusters))
        for cl in clusters:
            self._hl_metrics['max_cluster_size'] = max(self._hl_metrics['max_cluster_size'], len(cl))

        cache = hl_node.cluster_cache or {}
        forced_sig = tuple(sorted(forced_moves.items()))
        cached_confs: dict[tuple, Configuration] = {}
        missing: list[list[int]] = []
        for cluster in clusters:
            key = (tuple(cluster), window_used, forced_sig)
            conf = cache.get(key)
            if conf is not None and self._validate_configuration(hl_node.config, conf, graph, forced_moves):
                cached_confs[key] = conf
            else:
                missing.append(cluster)

        combined_pos: list[int] = [hl_node.config[aid] for aid in range(self.num_agents)]

        if not missing:
            for key, conf in cached_confs.items():
                cluster = list(key[0])
                for aid in cluster:
                    combined_pos[aid] = conf[aid]
            combined = Configuration(tuple(combined_pos))
            if not self._validate_configuration(hl_node.config, combined, graph, forced_moves):
                return None, None, None, False
            return combined, None, window_used, False

        states = self._build_cluster_states(
            hl_node=hl_node,
            chain=chain,
            forced_moves=forced_moves,
            clusters=missing,
        )
        hl_node.cluster_ll_states = states

        # Полностью растим LL-дерево каждого кластера
        for state in states:
            self._process_cluster_state(
                hl_node=hl_node,
                cluster_state=state,
                graph=graph,
                forced_moves=forced_moves,
            )
            if state.failed or state.found_config is None:
                # Фолбек целиком
                self._hl_metrics['cluster_fallbacks'] += 1
                return None, None, None, True
            # сохраняем найденные позиции кластера
            for aid in state.cluster:
                combined_pos[aid] = state.found_config[aid]
            key = (tuple(state.cluster), window_used, forced_sig)
            cache[key] = state.found_config

        combined = Configuration(tuple(combined_pos))
        if not self._validate_configuration(hl_node.config, combined, graph, forced_moves):
            self._hl_metrics['cluster_fallbacks'] += 1
            return None, None, None, True

        self._hl_metrics['cluster_successes'] += 1
        hl_node.cluster_cache = cache
        return combined, None, window_used, False

    # ------------------------------------------------------------
    # Кластерный LL-поиск
    # ------------------------------------------------------------
    def _build_cluster_states(
        self,
        hl_node: HLNode,
        chain: list[Constraint],
        forced_moves: dict[int, int],
        clusters: list[list[int]],
    ) -> list[ClusterLLState]:
        states: list[ClusterLLState] = []
        for cluster in clusters:
            extra_forced: list[tuple[int, int]] = []
            for aid in range(self.num_agents):
                if aid in cluster:
                    continue
                if aid in forced_moves:
                    continue
                extra_forced.append((aid, hl_node.config[aid]))
            root = self._clone_constraint_chain(chain, extra_forced)
            order = [aid for aid in hl_node.order if aid in cluster]
            state = ClusterLLState(
                cluster=list(cluster),
                order=order,
                constraint_tree=deque([(root, 0)]),  # (constraint, depth w.r.t cluster order)
            )
            states.append(state)
        return states

    def _expand_cluster_constraint(
        self,
        hl_node: HLNode,
        cluster_state: ClusterLLState,
        item: tuple[Constraint, int],
        pos_to_agent: dict[int, int],
        graph: GraphBase,
        forced_moves: dict[int, int],
    ) -> None:
        """Расширяем constraint для следующего агента кластера с учётом positive constraints."""
        constraint, depth = item
        if depth >= len(cluster_state.order):
            return
        agent_idx = cluster_state.order[depth]
        current_pos = hl_node.config[agent_idx]

        if agent_idx in forced_moves:
            next_vertices = [forced_moves[agent_idx]]
        else:
            next_vertices = list(graph.neighbors(current_pos))
            if not graph.is_blocked(current_pos) and current_pos not in next_vertices:
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

        for v in next_vertices:
            child = Constraint(
                parent=constraint,
                who=agent_idx,
                where=v,
                depth=constraint.depth + 1,
            )
            cluster_state.constraint_tree.append((child, depth + 1))
            self._hl_metrics['ll_nodes_created'] += 1
            self._hl_metrics['max_constraint_queue'] = max(
                self._hl_metrics['max_constraint_queue'],
                len(cluster_state.constraint_tree),
            )

    def _process_cluster_state(
        self,
        hl_node: HLNode,
        cluster_state: ClusterLLState,
        graph: GraphBase,
        forced_moves: dict[int, int],
    ) -> None:
        """Полностью обходит constraint_tree кластера, пока не найдёт конфигурацию или не исчерпает дерево."""
        pos_to_agent = {pos: idx for idx, pos in enumerate(hl_node.config.pos)}
        visited = 0
        while cluster_state.constraint_tree:
            visited += 1
            if visited > self.cluster_ll_limit:
                cluster_state.failed = True
                return
            constraint, depth = cluster_state.constraint_tree.popleft()
            self._hl_metrics['ll_expansions'] += 1
            # Расширяем детей, если не все агенты кластера назначены
            if depth < len(cluster_state.order):
                self._expand_cluster_constraint(
                    hl_node=hl_node,
                    cluster_state=cluster_state,
                    item=(constraint, depth),
                    pos_to_agent=pos_to_agent,
                    graph=graph,
                    forced_moves=forced_moves,
                )
                continue

            # Все агенты кластера зафиксированы — пробуем генератор
            conf = self.generator.generate(
                hl_node=hl_node,
                constraint=constraint,
                graph=graph,
                task_callback=self.task_callback,
                window=1,
                allow_goal_callback=False,
                agent_done=self._agent_done,
            )
            self._collect_generator_goal_events()
            if conf is None:
                continue
            if not self._validate_configuration(hl_node.config, conf, graph, forced_moves):
                continue
            cluster_state.found_config = conf
            return

        cluster_state.failed = True

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

    # ------------------------------------------------------------
    # Применение goal callback из генератора
    # ------------------------------------------------------------
    def _collect_generator_goal_events(self) -> None:
        """Забрать отложенные goal events из генератора (если он их умеет отдавать)."""
        if not hasattr(self.generator, "pop_goal_events"):
            return
        try:
            events = self.generator.pop_goal_events()  # type: ignore[attr-defined]
        except Exception:
            return
        if events:
            self._generator_goal_events.extend(events)

    def _apply_generator_goal_events(self, config: Configuration, verbose: bool) -> bool:
        """
        Если генератор уже получил новые цели внутри окна, применяем их,
        иначе используем стандартную _check_and_update_goals.
        """
        if not self._generator_goal_events:
            return self._check_and_update_goals(config, verbose)

        goals_updated = False
        merged: dict[int, tuple[int, int]] = {}
        for aid, old_goal, new_goal in self._generator_goal_events:
            merged[aid] = (old_goal, new_goal)

        for aid in range(self.num_agents):
            if self._agent_done[aid]:
                continue
            if aid in merged:
                continue
            if config[aid] != self.goals[aid]:
                self._goal_completion_ack[aid] = False

        for aid, (old_goal, new_goal) in merged.items():
            if self._agent_done[aid]:
                continue
            if config[aid] != self.goals[aid]:
                self._goal_completion_ack[aid] = False
                continue

            if not self._goal_completion_ack[aid]:
                self.completed_tasks_count[aid] += 1
                self.completed_tasks_history[aid].append(old_goal)
                self._goal_completion_ack[aid] = True

                if verbose:
                    print(f"    Агент {aid}: завершил цель {old_goal} "
                          f"(всего задач: {self.completed_tasks_count[aid]})")

            if self.max_tasks_per_agent is not None:
                if self.completed_tasks_count[aid] >= self.max_tasks_per_agent:
                    self._agent_done[aid] = True
                    continue

            if new_goal != old_goal:
                self.goals[aid] = new_goal
                self.goal_updates_count += 1
                self._goal_completion_ack[aid] = False
                goals_updated = True

                if verbose:
                    print(f"    Агент {aid}: новая цель {new_goal}")

        self._generator_goal_events = []
        return goals_updated
    
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

            if self.max_tasks_per_agent is not None:
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
        if self.max_tasks_per_agent is None:
            return False
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
