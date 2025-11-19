import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from core.graph.grid import GridGraph
from core.lifelong_lacam_integrated import LifelongLaCAMIntegrated
from strategies.generators.pibt import PIBTGenerator
from strategies.ordering.distance_ordering import DistanceOrdering
from strategies.open_policy.stack import StackOpen
from utils.kiva_loader import layout_to_grid, coords_to_indices

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
        assert len(set(nxt.pos)) == len(nxt.pos), f"Vertex collision на шаге {step}"
        for aid, (u, v) in enumerate(zip(cur.pos, nxt.pos)):
            assert v == u or v in graph.neighbors(u), f"Агент {aid} делает недопустимый ход на шаге {step}"
            assert not graph.is_blocked(v), f"Агент {aid} вошёл в стену на шаге {step}"
        for i in range(len(cur.pos)):
            for j in range(i + 1, len(cur.pos)):
                if cur.pos[i] == nxt.pos[j] and cur.pos[j] == nxt.pos[i] and cur.pos[i] != cur.pos[j]:
                    raise AssertionError(f"Edge swap между агентами {i} и {j} на шаге {step}")


def test_lifelong_kiva_large_tasks():
    data_path = Path("data/kiva_large_tasks.json")
    assert data_path.exists(), "Сгенерируйте задачи через scripts/generate_kiva_tasks.py"
    payload = json.loads(data_path.read_text())

    layout = payload["layout"]
    grid = layout_to_grid(layout)
    graph = GridGraph(grid)

    agents_to_use = 10
    starts_coords = payload["starts"][:agents_to_use]
    tasks_coords_list = payload["tasks"][:agents_to_use]
    print(starts_coords)
    starts = coords_to_indices(graph, starts_coords)
    tasks_list = [coords_to_indices(graph, coords) for coords in tasks_coords_list]
    assert len(starts) == len(tasks_list)
    tasks_per_agent = payload.get("tasks_per_agent", len(tasks_list[0]))
    assert tasks_per_agent >= 100, "Нужно минимум 100 задач на агента"

    tasks_runtime = [deque(agent_tasks) for agent_tasks in tasks_list]
    tasks_for_episodes = [deque(agent_tasks) for agent_tasks in tasks_list]
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
        max_tasks_per_agent=tasks_per_agent,
    )

    path = lacam.run(max_iterations=2_000_000, verbose=False)
    assert path is not None, "План не найден"
    _validate_plan(graph, path, starts)
    stats = lacam.get_statistics()
    assert all(x >= tasks_per_agent for x in stats["completed_tasks_per_agent"])

    if _VIS_EXPORT_FLAG:
        map_path, _ = export_visualizer_files("test_kiva_large_lifelong", graph, path)

        def split_into_episodes(configs: list, queues: list[deque]) -> list[list]:
            goals = [q[0] if q else None for q in queues]
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
        ep_paths = export_lifelong_episodes("test_kiva_large_lifelong", graph, episodes)
        vis_bin = shutil.which("mapf-visualizer-lifelong")
        if vis_bin and map_path and ep_paths:
            try:
                subprocess.run([vis_bin, map_path, *ep_paths], check=True)
            except Exception as exc:
                print(f"⚠️  Визуализатор не запустился: {exc}")


if __name__ == "__main__":
    test_lifelong_kiva_large_tasks()
