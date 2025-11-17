from __future__ import annotations
from typing import Optional, Dict, List, Set

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

    # ------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------
    def generate(
        self,
        hl_node: HLNode,
        constraint: Constraint,
        graph: GraphBase,
    ) -> Optional[Configuration]:

        old_conf: Configuration = hl_node.config
        num_agents = old_conf.num_agents()
        order: List[int] = hl_node.order

        # 1. Приоритеты агентов:
        #    pri[agent] = индекс в order; меньший индекс -> более высокий приоритет.
        pri: List[int] = [0] * num_agents
        for pr, ag in enumerate(order):
            pri[ag] = pr

        # 2. Собираем positive constraints из LL-цепочки
        forced_moves: Dict[int, int] = self._collect_constraints(constraint)

        # 3. Таблица занятости в старой конфигурации: pos -> agent
        pos2agent: Dict[int, int] = {}
        for agent_id, v in enumerate(old_conf.pos):
            pos2agent[v] = agent_id

        # 4. Массив новых позиций (результат одного шага)
        new_pos: List[Optional[int]] = [None] * num_agents

        # 5. Зарезервированные вершины в t+1
        reserved: Set[int] = set()

        # 6. "Стек" рекурсии для PIBT (для отсечения циклов)
        in_stack: Set[int] = set()

        # 7. Основной проход PIBT по приоритетам
        for agent in order:
            if new_pos[agent] is not None:
                # уже спланировали в рамках другого рекурсивного вызова
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
                # PIBT не смог найти план для этого агента (и рекурсивно
                # для тех, кого он пытался сдвинуть) → вся попытка неудачна.
                return None

        # 8. Проверка: все ли агенты спланированы
        for a in range(num_agents):
            if new_pos[a] is None:
                return None

        # 9. Дополнительная проверка edge-коллизий (на всякий случай)
        if self._has_edge_conflict(old_conf.pos, new_pos):
            return None

        # 10. Возвращаем новую конфигурацию
        return Configuration(tuple(new_pos))  # type: ignore[arg-type]

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

        cur_v = old_conf[agent]

        # 1) Формируем список кандидатов.
        #    Если есть constraint -> ровно один кандидат.
        if agent in forced_moves:
            candidates = [forced_moves[agent]]
        else:
            neigh = list(graph.neighbors(cur_v))
            # включаем stay
            if not graph.is_blocked(cur_v) and cur_v not in neigh:
                neigh.append(cur_v)
            # глупый порядок: как есть. Можно сортировать по дист. до цели.
            candidates = neigh

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
                # сам же и стоит в этой вершине → можно рассматривать как stay,
                # но сюда мы попадаем только если v == cur_v,
                # что обработано выше как свободный случай.
                continue

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
    # CONSTRAINT UTILS
    # ------------------------------------------------------------
    def _collect_constraints(self, constraint: Constraint) -> Dict[int, int]:
        """
        Собрать все positive constraints (who -> where) из цепочки
        LL-узлов, начиная от constraint и поднимаясь к корню.
        """
        forced: Dict[int, int] = {}
        node = constraint
        while node is not None and node.who is not None:
            forced[node.who] = node.where  # type: ignore[assignment]
            node = node.parent
        return forced

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
