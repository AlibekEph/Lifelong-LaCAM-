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
        
        # Explored таблица
        self._explored: Dict[Configuration, HLNode] = {}
        
        # Инициализация root node
        root_constraint = Constraint(parent=None, who=None, where=None, depth=0)
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
        iterations = 0
        
        while not self.open_policy.empty():
            if max_iterations is not None and iterations >= max_iterations:
                if verbose:
                    print(f"⚠️  Достигнут лимит итераций: {max_iterations}")
                break
            
            iterations += 1
            self.total_iterations = iterations
            
            if verbose and iterations % 5000 == 0:
                print(f"  Итерация {iterations}: Open={len(self.open_policy._stack) if hasattr(self.open_policy, '_stack') else '?'}, "
                      f"Explored={len(self._explored)}, "
                      f"Задач выполнено={sum(self.completed_tasks_count)}")
            
            hl_node = self.open_policy.peek()
            
            # Проверка: все ли агенты выполнили достаточно задач?
            if self._check_stopping_condition():
                if verbose:
                    print(f"\n✓ Все агенты выполнили по {self.max_tasks_per_agent} задач!")
                return hl_node.reconstruct_path()
            
            # Стандартная логика LaCAM
            if not hl_node.constraint_tree:
                self.open_policy.pop()
                continue
            
            ll_node = hl_node.constraint_tree.popleft()
            
            # Расширяем constraint tree
            if ll_node.depth < self.num_agents:
                agent_idx = hl_node.order[ll_node.depth]
                current_pos = hl_node.config[agent_idx]
                
                next_vertices: Iterable[int] = list(self.graph.neighbors(current_pos))
                if not self.graph.is_blocked(current_pos):
                    if current_pos not in next_vertices:
                        next_vertices = list(next_vertices) + [current_pos]
                
                for u in next_vertices:
                    child = Constraint(
                        parent=ll_node,
                        who=agent_idx,
                        where=u,
                        depth=ll_node.depth + 1,
                    )
                    hl_node.constraint_tree.append(child)
            
            # Генерируем новую конфигурацию
            new_config = self.generator.generate(
                hl_node=hl_node,
                constraint=ll_node,
                graph=self.graph,
            )
            
            if new_config is None:
                continue
            
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
            existing_node = self._explored.get(new_config)
            if existing_node is not None:
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
            
            # ВАЖНО: используем текущие (возможно обновлённые) цели
            new_order = self.ordering.reorder(
                config=new_config,
                prev_order=hl_node.order,
                parent=hl_node,
                graph=self.graph,
                goals=self.goals,  # ⬅️ Используем текущие цели!
            )
            
            child_node = HLNode(
                config=new_config,
                constraint_tree=child_tree,
                order=new_order,
                parent=hl_node,
            )
            
            self._explored[new_config] = child_node
            self.open_policy.push(child_node)
        
        # Не нашли решение
        if verbose:
            print(f"\n⚠️  Open пуст после {iterations} итераций")
            print(f"   Задач выполнено: {self.completed_tasks_count}")
        
        return None
    
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
            
            # Агент достиг своей цели?
            if current_pos == current_goal:
                # Назначаем новую цель через callback
                old_goal = current_goal
                new_goal = self.task_callback(agent_id, current_pos, old_goal)
                
                if new_goal != old_goal:
                    # Обновляем цель
                    self.goals[agent_id] = new_goal
                    
                    # Статистика
                    self.completed_tasks_count[agent_id] += 1
                    self.completed_tasks_history[agent_id].append(old_goal)
                    self.goal_updates_count += 1
                    
                    goals_updated = True
                    
                    if verbose:
                        print(f"    Агент {agent_id}: достиг {old_goal} → новая цель {new_goal} "
                              f"(всего задач: {self.completed_tasks_count[agent_id]})")
        
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
        return {
            'total_iterations': self.total_iterations,
            'goal_updates': self.goal_updates_count,
            'completed_tasks_per_agent': self.completed_tasks_count.copy(),
            'total_completed_tasks': sum(self.completed_tasks_count),
            'tasks_history': [tasks.copy() for tasks in self.completed_tasks_history],
        }

