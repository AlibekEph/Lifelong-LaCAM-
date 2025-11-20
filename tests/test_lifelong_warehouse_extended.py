import os
import sys
import shutil
import subprocess
import time
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from core.graph.grid import GridGraph
from core.lifelong_lacam_integrated import LifelongLaCAMIntegrated
from strategies.generators.pibt import PIBTGenerator
from strategies.ordering.distance_ordering import DistanceOrdering
from strategies.ordering.persistent_ordering import PersistentPriorityOrdering
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


def _pretty_print_metrics(stats: dict, label: str):
    prefix = f"[{label}] "
    hl = stats.get("hl_metrics") or {}
    runtime = hl.get("runtime_seconds")
    runtime_str = f"{runtime:.2f}s" if runtime is not None else "n/a"
    print(f"{prefix}HL metrics:")
    print(
        f"  runtime={runtime_str}, "
        f"nodes={hl.get('hl_nodes_created', 0)}, "
        f"revisited={hl.get('hl_revisited_nodes', 0)}, "
        f"iterations={hl.get('iterations', 0)}, "
        f"goal_updates={stats.get('goal_updates', 0)}"
    )
    print(
        f"  LL-nodes={hl.get('ll_nodes_created', 0)}, "
        f"max_queue={hl.get('max_constraint_queue', 0)}, "
        f"LL-expansions={hl.get('ll_expansions', 0)}, "
        f"generator={{success:{hl.get('generator_successes', 0)}, fail:{hl.get('generator_failures', 0)}}}"
    )

    gen = stats.get("generator_metrics") or {}
    stack = gen.get("stack_depth") or {}
    print(f"{prefix}PIBT metrics:")
    print(
        f"  generate={gen.get('generate_calls', 0)}, "
        f"pibt={{success:{gen.get('pibt_success', 0)}, fail:{gen.get('pibt_failures', 0)}, "
        f"time:{gen.get('pibt_time', 0.0):.2f}s}}, "
        f"bruteforce={{success:{gen.get('bruteforce_success', 0)}, fail:{gen.get('bruteforce_failures', 0)}, "
        f"time:{gen.get('bruteforce_time', 0.0):.2f}s}}"
    )
    if stack:
        print(
            f"  stack-depth: avg={stack.get('avg', 0):.2f}, "
            f"max={stack.get('max', 0)}, "
            f"p50={stack.get('p50', 0)}, "
            f"p90={stack.get('p90', 0)}, "
            f"p99={stack.get('p99', 0)}"
        )


def _make_shelf_grid(rows: int, cols: int, shelf_cols: list[int], gap_rows: set[int]) -> GridGraph:
    grid = np.zeros((rows, cols), dtype=bool)
    for r in range(1, rows - 1):
        for c in shelf_cols:
            if r not in gap_rows:
                grid[r, c] = True
    return GridGraph(grid)


def _build_branch_corridor(height: int = 21, width: int = 7, branch_len: int = 2):
    """
    Узкий вертикальный коридор с боковыми ответвлениями.
    Высота по умолчанию 21, ширина 7.
    """
    grid = np.ones((height, width), dtype=bool)
    main_col = width // 2
    for r in range(height):
        grid[r, main_col] = False

    candidate_rows = [2, 4, 6, 8, 10, 12, 14, 16, 18, 19]
    branch_rows = [row for row in candidate_rows if 0 < row < height - 1]
    branch_endpoints: list[tuple[int, int]] = []
    for idx, row in enumerate(branch_rows):
        direction = -1 if idx % 2 == 0 else 1
        for step in range(1, branch_len + 1):
            col = main_col + direction * step
            if 0 <= col < width:
                grid[row, col] = False
        end_col = max(0, min(width - 1, main_col + direction * branch_len))
        branch_endpoints.append((row, end_col))

    return GridGraph(grid), branch_rows, branch_endpoints, main_col


def test_lifelong_warehouse_six_agents():
    """
    Расширенный склад: 6 агентов, плотные полки, до 10 задач на агента.
    Проверяем отсутствие коллизий и завершение всех задач.
    """
    # Возвращаем стены: два ряда стеллажей с проходами (избегаем целей на самих стенах)
    graph = _make_shelf_grid(rows=11, cols=14, shelf_cols=[3, 8], gap_rows={3, 7})

    starts = [
        graph.to_idx(10, 0),
        graph.to_idx(10, 13),
        graph.to_idx(0, 0),
        graph.to_idx(0, 13),
        graph.to_idx(5, 2),
        graph.to_idx(5, 11),
    ]

    # Очереди умеренной длины (2 цели + возврат) вдоль коридоров между стеллажами
    tasks_coords = [
        [(9, 1), (9, 4), (9, 7), (9, 0), (8, 2)],
        [(9, 10), (9, 7), (9, 4), (9, 11), (9,1)],
        [(0, 1), (0, 4), (0, 7), (0, 0), (8, 1)],
        [(0, 10), (0, 7), (0, 4), (0, 11), (9,3)],
        [(5, 1), (5, 4), (5, 7), (5, 1), (9,2)],
        [(5, 10), (5, 7), (5, 4), (5, 10), (8,5)],
        
    ]

    tasks_runtime = [deque(graph.to_idx(r, c) for r, c in coords) for coords in tasks_coords]
    tasks_for_episodes = [deque(graph.to_idx(r, c) for r, c in coords) for coords in tasks_coords]
    initial_goals = [q[0] for q in tasks_runtime]

    def task_callback(agent_id: int, current_pos: int, old_goal: int) -> int:
        queue = tasks_runtime[agent_id]
        if queue and current_pos == queue[0]:
            queue.popleft()
        return queue[0] if queue else old_goal

    # Завершаем первые 2 задачи
    max_tasks = 4
    lacam = LifelongLaCAMIntegrated(
        graph=graph,
        starts=starts,
        initial_goals=initial_goals,
        generator=PIBTGenerator(),
        ordering=DistanceOrdering(),
        open_policy=StackOpen(),
        task_callback=task_callback,
        reinsert=True,
        max_tasks_per_agent=max_tasks,
    )

    start = time.time()
    path = lacam.run(max_iterations=6000000, verbose=False)
    duration = time.time() - start
    assert path is not None, "План не найден"
    _validate_plan(graph, path, starts)
    stats = lacam.get_statistics()
    print(f"[warehouse_six] solved in {duration:.2f}s")
    assert all(x >= max_tasks for x in stats["completed_tasks_per_agent"])
    _pretty_print_metrics(stats, label="warehouse_six")

    if _VIS_EXPORT_FLAG:
        map_path, _ = export_visualizer_files("test_lifelong_warehouse_six_agents", graph, path)

        def split_into_episodes(configs: list, queues: list[deque]) -> list[list]:
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
        ep_paths = export_lifelong_episodes("test_lifelong_warehouse_six_agents", graph, episodes)
        vis_bin = shutil.which("mapf-visualizer-lifelong")
        if vis_bin and map_path and ep_paths:
            try:
                subprocess.run([vis_bin, map_path, *ep_paths], check=True)
            except Exception as exc:
                print(f"⚠️  Визуализатор не запустился: {exc}")


def test_lifelong_dense_warehouse_eight_agents():
    """
    Сценарий с 8 агентами в компактном складе, до 10 задач на агента.
    """
    # Возвращаем стенки: три вертикальных стеллажа с проходами
    graph = _make_shelf_grid(rows=10, cols=12, shelf_cols=[2, 6, 9], gap_rows={2, 6})

    starts = [
        graph.to_idx(9, 0),
        graph.to_idx(9, 11),
        graph.to_idx(0, 0),
        graph.to_idx(0, 11),
        graph.to_idx(5, 1),
        graph.to_idx(5, 10),
        graph.to_idx(5, 4),
        graph.to_idx(5, 7),
    ]

    # Очереди средней длины (3 цели + возврат) вдоль коридоров
    tasks_coords = [
        [(9, 1), (9, 4), (9, 7), (9, 0), (0, 2)],
        [(9, 10), (9, 7), (9, 4), (9, 11), (4,1)],
        [(0, 1), (0, 4), (0, 7), (0, 0), (8, 1)],
        [(0, 10), (0, 7), (0, 4), (0, 11), (9,1)],
        [(5, 1), (5, 4), (5, 7), (5, 1), (6,2)],
        [(5, 10), (5, 7), (5, 4), (5, 10), (8,5)],
        [(5, 2), (5, 5), (5, 8), (5, 4), (9,5)],
        [(5, 9), (5, 6), (5, 3), (5, 7), (9,9)],
    ]

    tasks_runtime = [deque(graph.to_idx(r, c) for r, c in coords) for coords in tasks_coords]
    tasks_for_episodes = [deque(graph.to_idx(r, c) for r, c in coords) for coords in tasks_coords]
    initial_goals = [q[0] for q in tasks_runtime]

    def task_callback(agent_id: int, current_pos: int, old_goal: int) -> int:
        queue = tasks_runtime[agent_id]
        if queue and current_pos == queue[0]:
            queue.popleft()
        return queue[0] if queue else old_goal

    # Завершаем первые 3 задачи
    max_tasks = 3
    lacam = LifelongLaCAMIntegrated(
        graph=graph,
        starts=starts,
        initial_goals=initial_goals,
        generator=PIBTGenerator(),
        ordering=DistanceOrdering(),
        open_policy=StackOpen(),
        task_callback=task_callback,
        reinsert=True,
        max_tasks_per_agent=max_tasks,
    )

    start = time.time()
    path = lacam.run(max_iterations=400000, verbose=False)
    duration = time.time() - start
    assert path is not None, "План не найден"
    _validate_plan(graph, path, starts)
    stats = lacam.get_statistics()
    assert all(x >= max_tasks for x in stats["completed_tasks_per_agent"])
    print(f"[warehouse_eight] solved in {duration:.2f}s")
    _pretty_print_metrics(stats, label="warehouse_eight")

    if _VIS_EXPORT_FLAG:
        map_path, _ = export_visualizer_files("test_lifelong_dense_warehouse_eight_agents", graph, path)

        def split_into_episodes(configs: list, queues: list[deque]) -> list[list]:
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
        ep_paths = export_lifelong_episodes("test_lifelong_dense_warehouse_eight_agents", graph, episodes)
        vis_bin = shutil.which("mapf-visualizer-lifelong")
        if vis_bin and map_path and ep_paths:
            try:
                subprocess.run([vis_bin, map_path, *ep_paths], check=True)
            except Exception as exc:
                print(f"⚠️  Визуализатор не запустился: {exc}")


def test_lifelong_branch_corridor_parovoz():
    """
    Узкий коридор 21x7 с десятью агентами и 100 задачами на каждого.
    Заставляет агентов постоянно ездить туда-сюда по длинному коридору,
    что воспроизводит эффект "паровозиков".
    """
    graph, branch_rows, branch_endpoints, main_col = _build_branch_corridor()
    num_agents = len(branch_rows)
    assert num_agents == len(branch_endpoints) >= 10

    starts = [graph.to_idx(row, main_col) for row in branch_rows]
    starts = starts[:10]
    branch_endpoints = branch_endpoints[:10]
    num_agents = len(starts)

    top_idx = graph.to_idx(0, main_col)
    bottom_idx = graph.to_idx(graph.H - 1, main_col)
    tasks_per_agent = 100

    def build_tasks(aid: int) -> list[int]:
        own_branch = graph.to_idx(*branch_endpoints[aid])
        opposite = graph.to_idx(*branch_endpoints[(aid + num_agents // 2) % num_agents])
        pattern = [own_branch, top_idx, opposite, bottom_idx]
        seq: list[int] = []
        while len(seq) < tasks_per_agent:
            seq.extend(pattern)
        return seq[:tasks_per_agent]

    tasks_sequences = [build_tasks(aid) for aid in range(num_agents)]
    tasks_runtime = [deque(seq) for seq in tasks_sequences]
    tasks_for_episodes = [deque(seq) for seq in tasks_sequences]
    initial_goals = [seq[0] for seq in tasks_sequences]

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
        ordering=PersistentPriorityOrdering(),
        open_policy=StackOpen(),
        task_callback=task_callback,
        reinsert=True,
        max_tasks_per_agent=tasks_per_agent,
    )

    start_time = time.time()
    path = lacam.run(max_iterations=5_000_000, verbose=False)
    duration = time.time() - start_time
    assert path is not None, "План не найден"
    _validate_plan(graph, path, starts)
    stats = lacam.get_statistics()
    print(f"[corridor_parovoz] solved in {duration:.2f}s")
    _pretty_print_metrics(stats, label="corridor_parovoz")
    assert all(x >= tasks_per_agent for x in stats["completed_tasks_per_agent"])

    if _VIS_EXPORT_FLAG:
        map_path, _ = export_visualizer_files("test_lifelong_branch_corridor_parovoz", graph, path)

        def split_into_episodes(configs: list, queues: list[deque]) -> list[list]:
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
        ep_paths = export_lifelong_episodes("test_lifelong_branch_corridor_parovoz", graph, episodes)
        vis_bin = shutil.which("mapf-visualizer-lifelong")
        if vis_bin and map_path and ep_paths:
            try:
                subprocess.run([vis_bin, map_path, *ep_paths], check=True)
            except Exception as exc:
                print(f"⚠️  Визуализатор не запустился: {exc}")
