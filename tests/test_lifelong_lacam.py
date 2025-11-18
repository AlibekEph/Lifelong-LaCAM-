import os
import sys
import shutil
import subprocess
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from core.graph.grid import GridGraph
from core.lifelong_lacam_integrated import LifelongLaCAMIntegrated
from strategies.generators.pibt import PIBTGenerator
from strategies.ordering.distance_ordering import DistanceOrdering
from strategies.open_policy.stack import StackOpen


_VIS_EXPORT_ENV = os.environ.get("LACAM_VIS_EXPORT", "")
_VIS_EXPORT_FLAG = True if _VIS_EXPORT_ENV == "" else _VIS_EXPORT_ENV.lower() in {"1", "true", "yes", "on"}
_VIS_EXPORT_DIR = os.environ.get("LACAM_VIS_DIR", "vis_outputs")


def export_visualizer_files(name: str, graph: GridGraph, config_list: list):
    if not _VIS_EXPORT_FLAG:
        return None, None
    os.makedirs(_VIS_EXPORT_DIR, exist_ok=True)

    map_path = os.path.join(_VIS_EXPORT_DIR, f"{name}.map")
    sol_path = os.path.join(_VIS_EXPORT_DIR, f"{name}.txt")

    charset = {True: "@", False: "."}
    rows = ["".join(charset[bool(cell)] for cell in graph.grid[r, :]) for r in range(graph.H)]
    with open(map_path, "w", encoding="utf-8") as f:
        f.write("type octile\n")
        f.write(f"height {graph.H}\n")
        f.write(f"width {graph.W}\n")
        f.write("map\n")
        for row in rows:
            f.write(row + "\n")

    with open(sol_path, "w", encoding="utf-8") as f:
        for t, conf in enumerate(config_list):
            coords = []
            for pos in conf.pos:
                r, c = graph.to_rc(pos)
                coords.append(f"({c},{r})")
            f.write(f"{t}:" + ",".join(coords) + ",\n")
    return map_path, sol_path


def export_lifelong_episodes(name: str, graph: GridGraph, episodes: list[list]):
    """Экспортирует эпизоды Lifelong: ep1, ep2, ..."""
    if not _VIS_EXPORT_FLAG:
        return []
    os.makedirs(_VIS_EXPORT_DIR, exist_ok=True)
    ep_paths = []
    for idx, episode in enumerate(episodes, 1):
        ep_path = os.path.join(_VIS_EXPORT_DIR, f"{name}_ep{idx}.txt")
        with open(ep_path, "w", encoding="utf-8") as f:
            for t, conf in enumerate(episode):
                coords = []
                for pos in conf.pos:
                    r, c = graph.to_rc(pos)
                    coords.append(f"({c},{r})")
                f.write(f"{t}:" + ",".join(coords) + ",\n")
        ep_paths.append(ep_path)
    return ep_paths


def _validate_plan(graph: GridGraph, configs: list, starts: list[int]):
    assert configs, "План пуст"
    assert configs[0].pos == tuple(starts), "Начальная конфигурация неверна"

    for step, (cur, nxt) in enumerate(zip(configs, configs[1:]), 1):
        # вершинные коллизии
        assert len(set(nxt.pos)) == len(nxt.pos), f"Vertex collision на шаге {step}"
        for aid, (u, v) in enumerate(zip(cur.pos, nxt.pos)):
            assert v == u or v in graph.neighbors(u), f"Агент {aid} делает недопустимый ход на шаге {step}"
            assert not graph.is_blocked(v), f"Агент {aid} вошёл в стену на шаге {step}"
        # реберные коллизии
        for i in range(len(cur.pos)):
            for j in range(i + 1, len(cur.pos)):
                if cur.pos[i] == nxt.pos[j] and cur.pos[j] == nxt.pos[i] and cur.pos[i] != cur.pos[j]:
                    raise AssertionError(f"Edge swap между агентами {i} и {j} на шаге {step}")


def test_lifelong_with_obstacles_and_collisions():
    """
    Lifelong LaCAM: два агента с последовательными задачами, поле 5x5 без стен,
    цели заставляют агентов обмениваться углами (возможны коллизии по пути).
    """
    grid = np.zeros((5, 5), dtype=bool)
    graph = GridGraph(grid)

    # Стартовые позиции
    starts = [
        graph.to_idx(0, 0),  # агент 0, верхний левый
        graph.to_idx(4, 4),  # агент 1, нижний правый
    ]

    # Очерёдность целей для каждого агента (две задачи на агента)
    tasks = [
        deque([graph.to_idx(4, 0), graph.to_idx(0, 0)]),  # агент 0: вправо, потом назад
        deque([graph.to_idx(0, 4), graph.to_idx(4, 4)]),  # агент 1: влево, потом назад
    ]

    initial_goals = [tasks[i][0] for i in range(len(tasks))]

    def task_callback(agent_id: int, current_pos: int, old_goal: int) -> int:
        # Сдвигаем очередь, если агент достиг цели
        if tasks[agent_id] and current_pos == tasks[agent_id][0]:
            tasks[agent_id].popleft()
        return tasks[agent_id][0] if tasks[agent_id] else old_goal

    lacam = LifelongLaCAMIntegrated(
        graph=graph,
        starts=starts,
        initial_goals=initial_goals,
        generator=PIBTGenerator(),
        ordering=DistanceOrdering(),
        open_policy=StackOpen(),
        task_callback=task_callback,
        reinsert=False,
        max_tasks_per_agent=1,  # считаем выполненной первую задачу каждого агента
    )

    path = lacam.run(max_iterations=50000, verbose=True)
    assert path is not None, "План не найден"

    stats = lacam.get_statistics()
    print(f"\nМетрики Lifelong:"
          f" итераций={stats['total_iterations']},"
          f" обновлений целей={stats['goal_updates']},"
          f" выполнено задач={stats['completed_tasks_per_agent']}")

    _validate_plan(graph, path, starts)

    # обе первые задачи должны быть выполнены
    assert stats["completed_tasks_per_agent"] == [1, 1]


def test_lifelong_single_agent_sequences():
    """Один агент, две последовательные цели (останавливаемся после первой задачи), метрики."""
    # Карта 3x3 без стен
    grid = np.zeros((3, 3), dtype=bool)
    graph = GridGraph(grid)

    starts = [graph.to_idx(1, 1)]
    tasks = [deque([graph.to_idx(0, 0), graph.to_idx(2, 2)])]
    initial_goals = [tasks[0][0]]

    def task_callback(agent_id: int, current_pos: int, old_goal: int) -> int:
        if tasks[agent_id] and current_pos == tasks[agent_id][0]:
            tasks[agent_id].popleft()
        return tasks[agent_id][0] if tasks[agent_id] else old_goal

    lacam = LifelongLaCAMIntegrated(
        graph=graph,
        starts=starts,
        initial_goals=initial_goals,
        generator=PIBTGenerator(),
        ordering=DistanceOrdering(),
        open_policy=StackOpen(),
        task_callback=task_callback,
        reinsert=False,  # для одного агента без реинсерта работает стабильнее
        max_tasks_per_agent=1,  # считаем выполненной первую цель
    )

    path = lacam.run(max_iterations=10000, verbose=True)
    assert path is not None, "План не найден"

    stats = lacam.get_statistics()
    print(f"\nМетрики Single-agent:"
          f" итераций={stats['total_iterations']},"
          f" обновлений целей={stats['goal_updates']},"
          f" выполнено задач={stats['completed_tasks_per_agent']}")

    _validate_plan(graph, path, starts)
    assert stats["completed_tasks_per_agent"] == [1]


def test_lifelong_multi_agent_round_robin():
    """
    Lifelong MAPF с несколькими агентами и очередями задач (аналог warehouse picking):
    - Поле 4x4 без стен.
    - 3 агента стартуют на диагонали.
    - У каждого по две цели: дойти до противоположного угла, затем вернуться.
    Проверяем корректность плана и метрики.
    """
    grid = np.zeros((4, 4), dtype=bool)
    graph = GridGraph(grid)

    starts = [
        graph.to_idx(0, 0),  # агент 0
        graph.to_idx(1, 1),  # агент 1
        graph.to_idx(3, 3),  # агент 2
    ]

    tasks = [
        deque([graph.to_idx(3, 3), graph.to_idx(0, 0)]),
        deque([graph.to_idx(2, 2), graph.to_idx(1, 1)]),
        deque([graph.to_idx(0, 3), graph.to_idx(3, 3)]),
    ]
    initial_goals = [q[0] for q in tasks]

    def task_callback(agent_id: int, current_pos: int, old_goal: int) -> int:
        if tasks[agent_id] and current_pos == tasks[agent_id][0]:
            tasks[agent_id].popleft()
        return tasks[agent_id][0] if tasks[agent_id] else old_goal

    # Для жизнеспособности ограничимcя первой задачей каждого агента
    lacam = LifelongLaCAMIntegrated(
        graph=graph,
        starts=starts,
        initial_goals=initial_goals,
        generator=PIBTGenerator(),
        ordering=DistanceOrdering(),
        open_policy=StackOpen(),
        task_callback=task_callback,
        reinsert=True,
        max_tasks_per_agent=1,
    )

    path = lacam.run(max_iterations=50000, verbose=False)
    assert path is not None, "План не найден"

    stats = lacam.get_statistics()
    print(f"\nМетрики Round-robin:"
          f" итераций={stats['total_iterations']},"
          f" обновлений целей={stats['goal_updates']},"
          f" выполнено задач={stats['completed_tasks_per_agent']}")

    _validate_plan(graph, path, starts)
    assert stats["completed_tasks_per_agent"] == [1, 1, 1]


def test_lifelong_warehouse_four_agents():
    """
    Lifelong MAPF в условном складе:
    - 4 агента стартуют из углов.
    - У каждого по ~10 задач (плюс финальное возвращение на док для фиксации счётчика).
    - Сетка с «стеллажами» (вертикальные стены с проходами).
    - Проверяем отсутствие коллизий, метрики и экспорт визуализации.
    """
    grid = np.zeros((9, 12), dtype=bool)
    shelf_cols = [2, 5, 8]
    for r in range(1, 8):
        for c in shelf_cols:
            if r not in {3, 6}:  # проёмы для пересечения проходов
                grid[r, c] = True
    graph = GridGraph(grid)

    starts = [
        graph.to_idx(8, 0),   # агент 0: нижний левый угол
        graph.to_idx(8, 11),  # агент 1: нижний правый угол
        graph.to_idx(0, 0),   # агент 2: верхний левый угол
        graph.to_idx(0, 11),  # агент 3: верхний правый угол
    ]

    tasks_coords = [
        [
            (6, 1), (3, 1), (1, 1), (1, 4), (3, 4),
            (6, 4), (6, 7), (3, 7), (1, 7), (4, 10), (8, 0),
        ],
        [
            (6, 10), (3, 10), (1, 10), (1, 7), (3, 7),
            (6, 7), (6, 4), (3, 4), (1, 4), (4, 1), (8, 11),
        ],
        [
            (1, 1), (1, 4), (3, 1), (3, 4), (1, 7),
            (3, 7), (6, 1), (6, 4), (6, 7), (4, 10), (0, 0),
        ],
        [
            (1, 10), (1, 7), (3, 10), (3, 7), (1, 4),
            (3, 4), (6, 10), (6, 7), (6, 4), (4, 1), (0, 11),
        ],
    ]

    # Две копии очередей: одна для работы алгоритма, другая для построения эпизодов после.
    tasks_runtime = [deque(graph.to_idx(r, c) for r, c in coords) for coords in tasks_coords]
    tasks_for_episodes = [deque(graph.to_idx(r, c) for r, c in coords) for coords in tasks_coords]
    initial_goals = [q[0] for q in tasks_runtime]

    def task_callback(agent_id: int, current_pos: int, old_goal: int) -> int:
        queue = tasks_runtime[agent_id]
        if queue and current_pos == queue[0]:
            queue.popleft()
        return queue[0] if queue else old_goal

    lacam = LifelongLaCAMIntegrated(
        graph=graph,
        starts=starts,
        initial_goals=initial_goals,
        generator=PIBTGenerator(),
        ordering=DistanceOrdering(),
        open_policy=StackOpen(),
        task_callback=task_callback,
        reinsert=True,
        max_tasks_per_agent=10,
    )

    path = lacam.run(max_iterations=300000, verbose=False)
    assert path is not None, "План не найден"

    stats = lacam.get_statistics()
    _validate_plan(graph, path, starts)
    assert stats["completed_tasks_per_agent"] == [10, 10, 10, 10]

    # Экспорт карты/траектории и эпизодов для визуализатора
    map_path, _ = export_visualizer_files("test_lifelong_warehouse", graph, path)

    def split_into_episodes(configs: list, queues: list[deque]) -> list[list]:
        """Делит полный путь на эпизоды при смене цели любого агента."""
        goals = [q[0] for q in queues]
        episodes = []
        current_ep = [configs[0]]
        for conf in configs[1:]:
            current_ep.append(conf)
            updated = False
            for aid, q in enumerate(queues):
                if q and conf[aid] == q[0]:
                    q.popleft()
                    if q:
                        new_goal = q[0]
                        if new_goal != goals[aid]:
                            goals[aid] = new_goal
                            updated = True
            if updated:
                episodes.append(current_ep)
                current_ep = [conf]
        episodes.append(current_ep)
        return episodes

    episodes = split_into_episodes(path, tasks_for_episodes)
    ep_paths = export_lifelong_episodes("test_lifelong_warehouse", graph, episodes)

    # Автоматический запуск визуализатора (если установлен и экспорт включён)
    if _VIS_EXPORT_FLAG and map_path and ep_paths:
        vis_bin = shutil.which("mapf-visualizer-lifelong")
        if vis_bin is None:
            print("⚠️  mapf-visualizer-lifelong не найден в PATH, визуализация пропущена")
        else:
            cmd = [vis_bin, map_path, *ep_paths]
            try:
                subprocess.run(cmd, check=True)
            except Exception as exc:
                print(f"⚠️  Визуализатор не запустился: {exc}")
