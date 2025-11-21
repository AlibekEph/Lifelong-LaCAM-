from __future__ import annotations
from typing import Optional, Dict, List, Set, Optional as TypingOptional, Callable
from collections import Counter
from math import ceil
import time
import random

from core.configuration import Configuration
from core.constraint import Constraint
from core.hl_node import HLNode
from core.graph.base import GraphBase

from .base import ConfigGenerator


class PIBTGenerator(ConfigGenerator):
    """
    Реализация PIBT (Priority Inheritance with Backtracking) как генератора
    конфигураций для LaCAM.

    Делает РОВНО один шаг для всех агентов:
        config(t) -> config(t+1)

    Особенности:
    - Использует порядок агентов hl_node.order как базовый приоритет:
        чем раньше в списке, тем ВЫШЕ приоритет.
    - Учитывает positive constraints из LL-узла:
        для агента с constraint (who -> where) кандидаты = {where}.
    - Priority inheritance:
        если агент хочет пойти в вершину v, где стоит агент с меньшим
        приоритетом, рекурсивно пытаемся перепланировать этого агента.
    - Backtracking:
        если все кандидаты для агента (включая попытки наследования
        приоритета) проваливаются, функция возвращает None, и LaCAM
        продолжает поиск на high-level.
    """

    def __init__(self):
        # текущие цели агентов (может подменяться Lifelong LaCAM)
        self.current_goals: TypingOptional[List[int]] = None
        # глобальный счётчик тактов для круговой ротации приоритетов (PIBT)
        self._time_step: int = 0
        # накапливаем динамические приоритеты (кто дольше ждёт цели)
        self._priority_offsets: TypingOptional[List[float]] = None
        # собираем статистику работы генератора
        self._metrics = {
            "generate_calls": 0,
            "pibt_calls": 0,
            "pibt_success": 0,
            "pibt_failures": 0,
            "bruteforce_calls": 0,
            "bruteforce_success": 0,
            "bruteforce_failures": 0,
            "pibt_time": 0.0,
            "bruteforce_time": 0.0,
        }
        self._stack_depth_hist: Counter[int] = Counter()
        self._stack_depth_count: int = 0
        self._stack_depth_sum: int = 0
        self._stack_depth_max: int = 0
        # эвристические параметры
        self.occupancy_penalty: float = 5.0
        self.stay_bonus: float = 0.25
        # запоминаем последнюю позицию для приоритетного буста за "застревание"
        self._last_positions: TypingOptional[List[int]] = None
        # отложенные обновления целей, собранные внутри generate()
        self._pending_goal_events: list[tuple[int, int, int]] = []

    def set_current_goals(self, goals: List[int]) -> None:
        """Позволяет LaCAM/Lifelong передавать актуальные цели перед генерацией шага."""
        self.current_goals = goals

    # ------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------
    def generate(
        self,
        hl_node: HLNode,
        constraint: Constraint,
        graph: GraphBase,
        task_callback: TypingOptional[Callable[[int, int, int], int]] = None,
        window: int = 1,
        allow_goal_callback: bool = False,
        agent_done: TypingOptional[List[bool]] = None,
    ) -> Optional[Configuration]:
        self._metrics["generate_calls"] += 1
        self._pending_goal_events = []
        forced_moves, forced_order = self._collect_constraints(constraint)
        self._ensure_priority_vector(hl_node.config.num_agents())

        # 1) Быстрая попытка PIBT
        self._metrics["pibt_calls"] += 1
        start = time.perf_counter()
        conf = self._generate_pibt(hl_node, forced_moves, forced_order, graph)
        self._metrics["pibt_time"] += time.perf_counter() - start
        if conf is not None and self._is_valid_configuration(hl_node.config, conf, graph, forced_moves):
            self._metrics["pibt_success"] += 1
            self._update_priority_offsets(conf)
            self._capture_goal_events(
                conf,
                task_callback=task_callback,
                window=window,
                allow_goal_callback=allow_goal_callback,
                agent_done=agent_done,
            )
            return conf
        else:
            self._metrics["pibt_failures"] += 1

        # 2) Полный перебор даёт гарантированную полноту LL-шага
        self._metrics["bruteforce_calls"] += 1
        start = time.perf_counter()
        conf = self._generate_bruteforce(hl_node, forced_moves, forced_order, graph)
        self._metrics["bruteforce_time"] += time.perf_counter() - start
        if conf is not None and self._is_valid_configuration(hl_node.config, conf, graph, forced_moves):
            self._metrics["bruteforce_success"] += 1
            self._update_priority_offsets(conf)
            self._capture_goal_events(
                conf,
                task_callback=task_callback,
                window=window,
                allow_goal_callback=allow_goal_callback,
                agent_done=agent_done,
            )
            return conf
        else:
            self._metrics["bruteforce_failures"] += 1

        return None

    def get_metrics(self) -> Dict[str, int]:
        """Вернуть собранные метрики генератора."""
        metrics = dict(self._metrics)
        # средние времена PIBT/Bruteforce
        metrics["pibt_avg_time"] = (
            self._metrics["pibt_time"] / self._metrics["pibt_calls"]
            if self._metrics["pibt_calls"] > 0 else 0.0
        )
        metrics["bruteforce_avg_time"] = (
            self._metrics["bruteforce_time"] / self._metrics["bruteforce_calls"]
            if self._metrics["bruteforce_calls"] > 0 else 0.0
        )

        if self._stack_depth_count > 0:
            metrics["stack_depth"] = {
                "count": self._stack_depth_count,
                "avg": self._stack_depth_sum / self._stack_depth_count,
                "max": self._stack_depth_max,
                "p50": self._stack_percentile(0.5),
                "p90": self._stack_percentile(0.9),
                "p99": self._stack_percentile(0.99),
            }
        else:
            metrics["stack_depth"] = {
                "count": 0,
                "avg": 0,
                "max": 0,
                "p50": 0,
                "p90": 0,
                "p99": 0,
            }
        return metrics

    def _record_stack_depth(self, depth: int) -> None:
        self._stack_depth_hist[depth] += 1
        self._stack_depth_count += 1
        self._stack_depth_sum += depth
        if depth > self._stack_depth_max:
            self._stack_depth_max = depth

    def _stack_percentile(self, q: float) -> int:
        if self._stack_depth_count == 0:
            return 0
        threshold = ceil(q * self._stack_depth_count)
        cum = 0
        for depth in sorted(self._stack_depth_hist.keys()):
            cum += self._stack_depth_hist[depth]
            if cum >= threshold:
                return depth
        return self._stack_depth_max

    def _ensure_priority_vector(self, num_agents: int) -> None:
        if self._priority_offsets is None or len(self._priority_offsets) != num_agents:
            self._priority_offsets = [0.0 for _ in range(num_agents)]

    def _update_priority_offsets(self, new_conf: Configuration) -> None:
        if self.current_goals is None or self._priority_offsets is None:
            return
        if self._last_positions is None or len(self._last_positions) != len(new_conf.pos):
            self._last_positions = list(new_conf.pos)
        for aid, pos in enumerate(new_conf.pos):
            goal = self.current_goals[aid]
            if pos == goal:
                self._priority_offsets[aid] = 0.0
            else:
                stuck_bonus = 1.0 if self._last_positions[aid] == pos else 0.0
                self._priority_offsets[aid] += 1.0 + stuck_bonus
            self._last_positions[aid] = pos

    # ------------------------------------------------------------
    # INTERNAL: PIBT recursion
    # ------------------------------------------------------------
    def _pibt_dfs(
        self,
        agent: int,
        old_conf: Configuration,
        new_pos: List[Optional[int]],
        reserved: Set[int],
        forced_moves: Dict[int, int],
        pri: List[int],
        pos2agent: Dict[int, int],
        in_stack: Set[int],
        graph: GraphBase,
    ) -> bool:
        """
        Рекурсивная часть PIBT для одного агента.

        Гарантии:
        - Если функция возвращает True:
            new_pos[agent] установлен, reserved содержит его новую вершину.
        - Если функция возвращает False:
            new_pos и reserved остаются в том же состоянии, что и на входе
        """

        # Уже спланирован (другим рекурсивным вызовом)
        if new_pos[agent] is not None:
            return True

        # Защита от циклов priority inheritance
        if agent in in_stack:
            return False
        in_stack.add(agent)
        self._record_stack_depth(len(in_stack))

        cur_v = old_conf[agent]

        # 1) Формируем список кандидатов.
        #    Если есть constraint -> ровно один кандидат.
        if agent in forced_moves:
            candidates = [forced_moves[agent]]
        else:
            neigh = list(graph.neighbors(cur_v))
            random.shuffle(neigh)
            has_stay = False
            # включаем stay
            if not graph.is_blocked(cur_v) and cur_v not in neigh:
                has_stay = True
                #pass

            # сортировка по эвристике: ближе к цели → раньше
            if self.current_goals is not None:
                goal = self.current_goals[agent]

                def _heur(v: int) -> float:
                    d = graph.dist(v, goal)
                    if d < 0:
                        d = 10**9
                    penalty = 0.0
                    if v == cur_v:
                        penalty -= self.stay_bonus
                    else:
                        occ = pos2agent.get(v)
                        if occ is not None and (new_pos[occ] is None or new_pos[occ] == v):
                            penalty += self.occupancy_penalty
                        if v in reserved:
                            penalty += self.occupancy_penalty
                    return d + penalty

                # сортируем движущиеся кандидаты, stay добавим в конец
                move_candidates = sorted([v for v in neigh if v != cur_v], key=_heur)
                candidates = move_candidates + ([cur_v] if has_stay else [])
            else:
                move_candidates = [v for v in neigh if v != cur_v]
                candidates = move_candidates + ([cur_v] if has_stay else [])

        # 2) Пытаемся каждый кандидат
        for v in candidates:
            # сразу отбрасываем запрещённые вершины
            if graph.is_blocked(v):
                continue

            # Проверяем, не зарезервирована ли вершина кем-то другим
            # (если резервировал другой агент, не текущий occupant)
            if v in reserved:
                # возможно это "occupant", который уже перепланирован;
                # но тогда pos2agent[v] либо другой, либо уже не на v.
                occ = pos2agent.get(v)
                if occ is None or new_pos[occ] == v:
                    # реально занято на t+1
                    continue

            # Определяем агента, который стоит в v на старой конфигурации
            occupant = pos2agent.get(v, None)
            if occupant is not None:
                # если occupant уже уходит из v (new_pos другое) → v свободно
                if new_pos[occupant] is not None and new_pos[occupant] != v:
                    occupant = None

            # 2.1. Если вершина свободна на t+1 → просто занимаем её
            if occupant is None:
                if self._check_edge_conflicts_local(agent, cur_v, v, old_conf.pos, new_pos):
                    # edge conflict (swap) → пропускаем этот v
                    continue

                # Регистрируем ход и выходим с успехом
                new_pos[agent] = v
                reserved.add(v)
                in_stack.remove(agent)
                return True

            # 2.2. Вершина занята другим агентом в старой конфигурации.
            #      Пробуем priority inheritance (если наш приоритет выше).
            if occupant == agent:
                # stay на месте: разрешаем его, если нет edge-конфликта
                if self._check_edge_conflicts_local(agent, cur_v, v, old_conf.pos, new_pos):
                    in_stack.remove(agent)
                    return False

                new_pos[agent] = v
                reserved.add(v)
                in_stack.remove(agent)
                return True

            # Если наш приоритет НЕ выше → не можем "пинать" occupant.
            if pri[agent] >= pri[occupant]:
                continue

            # Если occupant уже в рекурсивном стеке → цикл, пропускаем
            if occupant in in_stack:
                continue

            # Пытаемся перепланировать occupant
            if not self._pibt_dfs(
                agent=occupant,
                old_conf=old_conf,
                new_pos=new_pos,
                reserved=reserved,
                forced_moves=forced_moves,
                pri=pri,
                pos2agent=pos2agent,
                in_stack=in_stack,
                graph=graph,
            ):
                # Не удалось сдвинуть occupant → пробуем следующий кандидат v
                continue

            # После успешного перепланирования occupant:
            # - new_pos[occupant] установлено
            # - reserved содержит его новую вершину
            # Если он всё равно остался в v → мы всё ещё не можем занять v.
            if new_pos[occupant] == v:
                continue

            # Проверяем edge-конфликт (swap) с уже запланированными агентами
            if self._check_edge_conflicts_local(agent, cur_v, v, old_conf.pos, new_pos):
                continue

            # Теперь v свободна → занимаем её
            new_pos[agent] = v
            reserved.add(v)
            in_stack.remove(agent)
            return True

        # Если ни один кандидат не сработал → откат:
        # new_pos[agent] так и не был установлен, reserved не меняли
        in_stack.remove(agent)
        return False

    # ------------------------------------------------------------
    # PIBT core (выделено для возможности fallback)
    # ------------------------------------------------------------
    def _generate_pibt(
        self,
        hl_node: HLNode,
        forced_moves: Dict[int, int],
        forced_order: List[int],
        graph: GraphBase,
    ) -> Optional[Configuration]:
        old_conf: Configuration = hl_node.config
        num_agents = old_conf.num_agents()
        order: List[int] = hl_node.order

        base_rank: Dict[int, int] = {}
        for idx, ag in enumerate(order):
            base_rank[ag] = idx
        forced_index: Dict[int, int] = {ag: idx for idx, ag in enumerate(forced_order)}

        priority_offsets = self._priority_offsets or [0.0 for _ in range(num_agents)]
        forced_bonus = num_agents + len(forced_index)
        effective_priority: List[float] = [0.0] * num_agents
        for ag in range(num_agents):
            rank = base_rank.get(ag, num_agents + ag)
            eff = rank - priority_offsets[ag]
            idx = forced_index.get(ag)
            if idx is not None:
                eff -= (forced_bonus - idx)
            effective_priority[ag] = eff

        rotated = sorted(
            list(range(num_agents)),
            key=lambda ag: (effective_priority[ag], base_rank.get(ag, num_agents + ag), ag),
        )

        pri: List[int] = [0] * num_agents
        for pr, ag in enumerate(rotated):
            pri[ag] = pr

        pos2agent: Dict[int, int] = {}
        for agent_id, v in enumerate(old_conf.pos):
            pos2agent[v] = agent_id

        new_pos: List[Optional[int]] = [None] * num_agents
        reserved: Set[int] = set()
        in_stack: Set[int] = set()

        for agent in rotated:
            if new_pos[agent] is not None:
                continue
            if not self._pibt_dfs(
                agent=agent,
                old_conf=old_conf,
                new_pos=new_pos,
                reserved=reserved,
                forced_moves=forced_moves,
                pri=pri,
                pos2agent=pos2agent,
                in_stack=in_stack,
                graph=graph,
            ):
                return None

        for a in range(num_agents):
            if new_pos[a] is None:
                return None

        if self._has_edge_conflict(old_conf.pos, new_pos):
            return None

        # завершили попытку LL-шага → увеличиваем глобальный таймер
        self._time_step += 1

        return Configuration(tuple(new_pos))  # type: ignore[arg-type]

    def _capture_goal_events(
        self,
        conf: Configuration,
        task_callback: TypingOptional[Callable[[int, int, int], int]],
        window: int,
        allow_goal_callback: bool,
        agent_done: TypingOptional[List[bool]],
    ) -> None:
        """
        Если работаем в оконном режиме (window>1) и нам разрешили
        (allow_goal_callback), заранее получаем новые цели через callback.
        Это нужно, чтобы генератор мог знать новые цели в пределах окна,
        но сами обновления передаются наружу через _pending_goal_events.
        """
        if task_callback is None or not allow_goal_callback or window <= 1:
            return
        if self.current_goals is None:
            return
        events: list[tuple[int, int, int]] = []
        for aid, pos in enumerate(conf.pos):
            if agent_done is not None and aid < len(agent_done) and agent_done[aid]:
                continue
            goal = self.current_goals[aid]
            if pos != goal:
                continue
            new_goal = task_callback(aid, pos, goal)
            events.append((aid, goal, new_goal))
            if new_goal != goal:
                self.current_goals[aid] = new_goal
        self._pending_goal_events = events

    def pop_goal_events(self) -> list[tuple[int, int, int]]:
        """Забрать и очистить собранные goal events (aid, old_goal, new_goal)."""
        events = self._pending_goal_events
        self._pending_goal_events = []
        return events

    # ------------------------------------------------------------
    # Полный перебор одного шага (stay+соседи) для всех агентов
    # ------------------------------------------------------------
    def _generate_bruteforce(
        self,
        hl_node: HLNode,
        forced: Dict[int, int],
        forced_order: List[int],
        graph: GraphBase,
    ) -> Optional[Configuration]:
        old_conf: Configuration = hl_node.config
        num_agents = old_conf.num_agents()

        # список кандидатов на ход для каждого агента
        cand_lists: List[List[int]] = [[] for _ in range(num_agents)]
        for aid in range(num_agents):
            cur = old_conf[aid]
            if aid in forced:
                cand_lists[aid] = [forced[aid]]
            else:
                neigh = list(graph.neighbors(cur))
                if not graph.is_blocked(cur) and cur not in neigh:
                    neigh.append(cur)
                # приоритет по дистанции до цели, если есть
                if self.current_goals is not None:
                    goal = self.current_goals[aid]
                    def _heur(v: int) -> float:
                        d = graph.dist(v, goal)
                        if d < 0:
                            d = 10**9
                        if v == cur:
                            d += 0.5
                        return d

                    neigh.sort(key=_heur)
                cand_lists[aid] = neigh

        forced_seen: Set[int] = set()
        order_seq: List[int] = []
        for aid in forced_order:
            if aid not in forced_seen:
                order_seq.append(aid)
                forced_seen.add(aid)
        for aid in hl_node.order:
            if aid not in forced_seen:
                order_seq.append(aid)
                forced_seen.add(aid)
        for aid in range(num_agents):
            if aid not in forced_seen:
                order_seq.append(aid)
                forced_seen.add(aid)

        # DFS по комбинациям
        new_pos = [None] * num_agents
        used = set()

        def dfs(idx: int) -> bool:
            if idx == len(order_seq):
                # edge-конфликты
                if self._has_edge_conflict(old_conf.pos, new_pos):
                    return False
                return True
            agent = order_seq[idx]
            for v in cand_lists[agent]:
                if graph.is_blocked(v):
                    continue
                if v in used:
                    continue
                # edge-свап с уже размещёнными
                conflict = False
                cur_pos = old_conf.pos[agent]
                for j in range(idx):
                    other_agent = order_seq[j]
                    other_new = new_pos[other_agent]
                    if other_new is None:
                        continue
                    if cur_pos == other_new and old_conf.pos[other_agent] == v and cur_pos != old_conf.pos[other_agent]:
                        conflict = True
                        break
                if conflict:
                    continue
                used.add(v)
                new_pos[agent] = v
                if dfs(idx + 1):
                    return True
                used.remove(v)
                new_pos[agent] = None
            return False

        if dfs(0):
            return Configuration(tuple(new_pos))  # type: ignore[arg-type]
        return None

    def _is_valid_configuration(
        self,
        old_conf: Configuration,
        new_conf: Configuration,
        graph: GraphBase,
        forced_moves: Dict[int, int],
    ) -> bool:
        """
        Проверка, что найденная конфигурация действительно удовлетворяет
        LL-ограничениям: нет вершинных/ребровых конфликтов, ходы допустимы,
        соблюдены положительные ограничения.
        """
        # vertex collisions
        if len(set(new_conf.pos)) != len(new_conf.pos):
            return False

        # допустимость ходов и positive constraints
        for aid, (u, v) in enumerate(zip(old_conf.pos, new_conf.pos)):
            if graph.is_blocked(v):
                return False
            if v != u and v not in graph.neighbors(u):
                return False
            forced = forced_moves.get(aid)
            if forced is not None and forced != v:
                return False

        # edge swaps
        return not self._has_edge_conflict(old_conf.pos, list(new_conf.pos))

    # ------------------------------------------------------------
    # CONSTRAINT UTILS
    # ------------------------------------------------------------
    def _collect_constraints(self, constraint: Constraint) -> tuple[Dict[int, int], List[int]]:
        """
        Собрать все positive constraints (who -> where) из цепочки
        LL-узлов, начиная от constraint и поднимаясь к корню.
        """
        forced: Dict[int, int] = {}
        order: List[int] = []
        node_chain: List[Constraint] = []
        node = constraint
        while node is not None and node.who is not None:
            node_chain.append(node)
            node = node.parent
        for constrained in reversed(node_chain):
            forced[constrained.who] = constrained.where  # type: ignore[assignment]
            order.append(constrained.who)
        return forced, order

    # ------------------------------------------------------------
    # EDGE CONFLICT CHECKS
    # ------------------------------------------------------------
    def _has_edge_conflict(
        self,
        old_pos: tuple[int, ...],
        new_pos: List[Optional[int]],
    ) -> bool:
        """
        Глобальная проверка edge-коллизий:
        запрещаем ситуацию, когда два агента i,j меняются местами:
            old[i] == new[j] and old[j] == new[i]
        """
        n = len(old_pos)
        for i in range(n):
            if new_pos[i] is None:
                continue
            for j in range(i + 1, n):
                if new_pos[j] is None:
                    continue
                if old_pos[i] == new_pos[j] and old_pos[j] == new_pos[i] \
                        and old_pos[i] != old_pos[j]:
                    return True
        return False

    def _check_edge_conflicts_local(
        self,
        agent: int,
        cur_v: int,
        next_v: int,
        old_pos: tuple[int, ...],
        new_pos: List[Optional[int]],
    ) -> bool:
        """
        Локальная проверка edge-конфликта для одного кандидата (agent: cur_v -> next_v):
        проверяем уже спланированных агентов j:
            old[j] == next_v и new_pos[j] == cur_v
        """
        if next_v == cur_v:
            # stay не может вызвать ребровой перестановки
            return False

        for j, qj in enumerate(new_pos):
            if qj is None or j == agent:
                continue
            if old_pos[j] == next_v and qj == cur_v:
                return True
        return False
