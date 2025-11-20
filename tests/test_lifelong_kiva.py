import os
import sys
import json
import shutil
import subprocess
import argparse
from pathlib import Path
from collections import deque
from typing import Optional
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from core.graph.grid import GridGraph
from core.lifelong_lacam_integrated import LifelongLaCAMIntegrated
from strategies.generators.pibt import PIBTGenerator
from strategies.ordering.persistent_ordering import PersistentPriorityOrdering
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


def _pretty_print_metrics(stats: dict, label: str = ""):
    prefix = f"[{label}] " if label else ""
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
        f"time:{gen.get('pibt_time', 0.0):.2f}s, avg:{gen.get('pibt_avg_time', 0.0):.4f}s}}, "
        f"bruteforce={{success:{gen.get('bruteforce_success', 0)}, fail:{gen.get('bruteforce_failures', 0)}, "
        f"time:{gen.get('bruteforce_time', 0.0):.2f}s, avg:{gen.get('bruteforce_avg_time', 0.0):.4f}s}}"
    )
    if stack:
        print(
            f"  stack-depth: avg={stack.get('avg', 0):.2f}, "
            f"max={stack.get('max', 0)}, "
            f"p50={stack.get('p50', 0)}, "
            f"p90={stack.get('p90', 0)}, "
            f"p99={stack.get('p99', 0)}"
        )


def _repeat_to_length(seq: list[int], target: int) -> list[int]:
    if not seq:
        raise ValueError("Empty task sequence")
    if len(seq) >= target:
        return seq[:target]
    repeated: list[int] = []
    while len(repeated) < target:
        repeated.extend(seq)
    return repeated[:target]


def test_lifelong_kiva_large_tasks(
    num_agents: int = 10,
    enable_clustering: bool = True,
    tasks_per_agent_override: Optional[int] = None,
):
    data_path = Path("data/kiva_large_tasks.json")
    assert data_path.exists(), "Сгенерируйте задачи через scripts/generate_kiva_tasks.py"
    payload = json.loads(data_path.read_text())

    layout = payload["layout"]
    grid = layout_to_grid(layout)
    graph = GridGraph(grid)

    agents_to_use = num_agents
    starts_coords = payload["starts"][:agents_to_use]
    tasks_coords_list = payload["tasks"][:agents_to_use]
    starts = coords_to_indices(graph, starts_coords)
    base_tasks = [coords_to_indices(graph, coords) for coords in tasks_coords_list]
    assert len(starts) == len(base_tasks)
    tasks_per_agent = tasks_per_agent_override if tasks_per_agent_override is not None else payload.get(
        "tasks_per_agent",
        len(base_tasks[0]),
    )
    assert tasks_per_agent > 0, "Число задач на агента должно быть > 0"
    print(f"tasks_per_agent={tasks_per_agent}")

    tasks_list = [_repeat_to_length(agent_tasks, tasks_per_agent) for agent_tasks in base_tasks]
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
        ordering=PersistentPriorityOrdering(1, 3),
        open_policy=StackOpen(),
        task_callback=task_callback,
        reinsert=True,
        max_tasks_per_agent=tasks_per_agent,
        enable_clustering=enable_clustering,
    )

    start = time.time()
    path = lacam.run(max_iterations=2_000_000, verbose=False)
    duration = time.time() - start
    assert path is not None, "План не найден"
    _validate_plan(graph, path, starts)
    stats = lacam.get_statistics()
    print(f"[kiva] solved in {duration:.2f}s")
    assert all(x >= tasks_per_agent for x in stats["completed_tasks_per_agent"])
    _pretty_print_metrics(stats, label="kiva")

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
    parser = argparse.ArgumentParser(description="Тест lifelong MAPF на Kiva layout")
    parser.add_argument(
        "--num-agents",
        type=int,
        default=10,
        help="Количество агентов для запуска (по умолчанию: 10)"
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Тихий режим (сокращённый вывод)"
    )
    parser.add_argument(
        "--no-clustering",
        action="store_true",
        help="Отключить кластеризацию LL-шага",
    )
    parser.add_argument(
        "--tasks-per-agent",
        type=int,
        default=None,
        help="Количество задач на агента (override payload, по умолчанию берётся из kiva_large_tasks.json)",
    )
    args = parser.parse_args()
    lacam_clustering = not args.no_clustering
    test_lifelong_kiva_large_tasks(
        num_agents=args.num_agents,
        enable_clustering=lacam_clustering,
        tasks_per_agent_override=args.tasks_per_agent,
    )
