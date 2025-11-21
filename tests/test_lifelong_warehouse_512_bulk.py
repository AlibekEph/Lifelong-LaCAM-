import os
import sys
import json
import argparse
import time
import shutil
import subprocess
from pathlib import Path
from collections import deque
import random
import numpy as np

# Чтобы исполнять как сценарий, добавляем src в sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.graph.grid import GridGraph
from core.lifelong_lacam_integrated import LifelongLaCAMIntegrated
from core.configuration import Configuration
from strategies.generators.pibt import PIBTGenerator
from strategies.ordering.persistent_ordering import PersistentPriorityOrdering
from strategies.open_policy.stack import StackOpen
from strategies.open_policy.completed_priority import CompletedPriorityOpen
from utils.kiva_loader import layout_to_grid, generate_kiva_tasks

WAREHOUSE_MAP_PATH = Path("data/warehouse_512_map.json")

_VIS_EXPORT_ENV = os.environ.get("LACAM_VIS_EXPORT", "")
_VIS_EXPORT_FLAG = True if _VIS_EXPORT_ENV == "" else _VIS_EXPORT_ENV.lower() in {"1", "true", "yes", "on"}
_VIS_EXPORT_DIR = os.environ.get("LACAM_VIS_DIR", "vis_outputs")


def export_visualizer_files(name: str, graph: GridGraph, config_list: list[Configuration]):
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


def sample_unique_free(graph: GridGraph, num: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    free = [idx for idx in range(graph.num_vertices()) if not graph.is_blocked(idx)]
    assert len(free) >= num, "Недостаточно свободных клеток для стартов"
    return rng.sample(free, num)


def total_moves(path: list[Configuration]) -> int:
    moves = 0
    for cur, nxt in zip(path, path[1:]):
        for u, v in zip(cur.pos, nxt.pos):
            if u != v:
                moves += 1
    return moves


def run_lacam(
    graph: GridGraph,
    starts: list[int],
    tasks: list[list[int]],
    stop_mode: str,
    stop_value: int,
    enable_clustering: bool,
    use_priority_open: bool,
    cluster_window: int,
):
    base_tasks = [list(agent_tasks) if agent_tasks else [] for agent_tasks in tasks]
    tasks_runtime = [deque(agent_tasks) for agent_tasks in tasks]
    max_tasks_per_agent = stop_value if stop_mode == "tasks" else max(len(t) for t in tasks)
    initial_goals = [q[0] for q in tasks_runtime]

    def task_callback(agent_id: int, current_pos: int, old_goal: int) -> int:
        queue = tasks_runtime[agent_id]
        if queue and current_pos == queue[0]:
            queue.popleft()
        if not queue and base_tasks[agent_id]:
            # повторяем исходную последовательность, чтобы всегда были цели
            queue.extend(base_tasks[agent_id])
        return queue[0] if queue else old_goal

    open_policy = CompletedPriorityOpen() if use_priority_open else StackOpen()

    lacam = LifelongLaCAMIntegrated(
        graph=graph,
        starts=starts,
        initial_goals=initial_goals,
        generator=PIBTGenerator(),
        ordering=PersistentPriorityOrdering(1, 3),
        open_policy=open_policy,
        task_callback=task_callback,
        reinsert=True,
        max_tasks_per_agent=max_tasks_per_agent,
        enable_clustering=enable_clustering,
        cluster_window_w=cluster_window,
    )

    t0 = time.time()
    max_iters = stop_value if stop_mode == "ticks" else 5_000_000
    path = lacam.run(max_iterations=max_iters, verbose=False)
    if path is None:
        try:
            node = lacam.open_policy.peek()  # type: ignore[attr-defined]
            path = node.reconstruct_path()
        except Exception:
            path = None
    runtime = time.time() - t0
    if path is None:
        return {
            "ticks": max_iters,
            "total_moves": None,
            "completed_tasks": lacam.get_statistics().get("completed_tasks_per_agent"),
            "runtime": runtime,
            "note": "path not found before stop",
            "path": None,
        }
    ticks = len(path) - 1

    # Детализированные метрики для сверки с реализацией на C++
    print(f"\n=== Python Metrics ===", file=sys.stderr)
    print(f"Iterations: {lacam._hl_metrics.get('iterations', 0)}", file=sys.stderr)
    print(f"LL expansions: {lacam._hl_metrics.get('ll_expansions', 0)}", file=sys.stderr)
    print(f"LL nodes created: {lacam._hl_metrics.get('ll_nodes_created', 0)}", file=sys.stderr)
    print(f"Generator successes: {lacam._hl_metrics.get('generator_successes', 0)}", file=sys.stderr)
    print(f"Generator failures: {lacam._hl_metrics.get('generator_failures', 0)}", file=sys.stderr)
    print(f"HL revisited nodes: {lacam._hl_metrics.get('hl_revisited_nodes', 0)}", file=sys.stderr)
    print(f"HL nodes created: {lacam._hl_metrics.get('hl_nodes_created', 0)}", file=sys.stderr)

    return {
        "ticks": ticks,
        "total_moves": total_moves(path),
        "completed_tasks": lacam.get_statistics().get("completed_tasks_per_agent"),
        "runtime": runtime,
        "note": None,
        "path": path,
    }


def run_pypibt(
    graph: GridGraph,
    starts: list[int],
    tasks: list[list[int]],
    stop_mode: str,
    stop_value: int,
    seed: int,
):
    try:
        from pypibt import PIBT
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "extern" / "pypibt" / "src"))
        import typing
        if not hasattr(typing, "TypeAlias"):
            from typing import Any
            typing.TypeAlias = Any  # type: ignore[attr-defined]
        from pypibt import PIBT  # type: ignore

    free_grid = (~graph.grid).astype(bool)  # True=free для pypibt
    positions = list(starts)
    goals = [agent_tasks[0] for agent_tasks in tasks]
    remaining = [len(agent_tasks) for agent_tasks in tasks]
    tasks_runtime = [deque(agent_tasks) for agent_tasks in tasks]
    path = [Configuration(tuple(positions))]
    ticks = 0
    total_mv = 0
    t0 = time.time()

    def idx_to_rc(idx: int) -> tuple[int, int]:
        return graph.to_rc(idx)

    def rc_to_idx(rc: tuple[int, int]) -> int:
        r, c = rc
        return graph.to_idx(r, c)

    max_iters = stop_value if stop_mode == "ticks" else 5_000_000
    while ticks < max_iters:
        positions_rc = [idx_to_rc(p) for p in positions]
        goals_rc = [idx_to_rc(g) for g in goals]
        pibt = PIBT(free_grid, positions_rc, goals_rc, seed=seed + ticks)
        priorities = [pibt.dist_tables[i].get(positions_rc[i]) / free_grid.size for i in range(len(positions))]
        next_rc = pibt.step(positions_rc, priorities)
        next_idx = [rc_to_idx(rc) for rc in next_rc]

        total_mv += sum(1 for u, v in zip(positions, next_idx) if u != v)
        ticks += 1
        positions = next_idx
        path.append(Configuration(tuple(positions)))

        for aid, pos in enumerate(positions):
            if not tasks_runtime[aid]:
                continue
            if pos == goals[aid]:
                tasks_runtime[aid].popleft()
                remaining[aid] -= 1
                if remaining[aid] > 0:
                    goals[aid] = tasks_runtime[aid][0]
        if stop_mode == "tasks" and all(r <= 0 for r in remaining):
            break

    runtime = time.time() - t0
    return {
        "ticks": ticks,
        "total_moves": total_mv,
        "completed_tasks": [len(tasks[aid]) - max(r, 0) for aid, r in enumerate(remaining)],
        "runtime": runtime,
        "note": None if (stop_mode != "tasks" or all(r <= 0 for r in remaining)) else "stopped by tick limit",
        "path": path,
    }


def main():
    parser = argparse.ArgumentParser(description="Массовый тест на складе 512x512 (warehouse layout)")
    parser.add_argument("--map-json", type=Path, default=WAREHOUSE_MAP_PATH, help="Путь до json с раскладкой склада")
    parser.add_argument("--num-agents", type=int, default=256, help="Количество агентов")
    parser.add_argument("--tasks-per-agent", type=int, default=400, help="Число задач на агента")
    parser.add_argument("--seed", type=int, default=7, help="Сид генерации стартов/задач")
    parser.add_argument("--solver", choices=["lacam", "pypibt"], default="lacam", help="Выбор решателя")
    parser.add_argument("--stop-mode", choices=["tasks", "ticks"], default="tasks", help="Правило остановки")
    parser.add_argument("--ticks-limit", type=int, default=20000, help="Лимит тиков при stop-mode=ticks")
    parser.add_argument("--no-clustering", action="store_true", help="Отключить кластеризацию (для lacam)")
    parser.add_argument("--priority-open", action="store_true", help="Приоритизировать HL-узлы по числу выполненных задач")
    parser.add_argument("--cluster-window", type=int, default=2, help="Максимальная длина окна для кластеризации (w)")
    parser.add_argument("--min-goal-distance", type=int, default=6, help="Минимальное расстояние между целями подряд")
    args = parser.parse_args()

    assert args.map_json.exists(), f"Не найден файл карты: {args.map_json}"
    payload = json.loads(args.map_json.read_text())
    grid = layout_to_grid(payload["layout"])
    graph = GridGraph(grid)

    random.seed(args.seed)
    np.random.seed(args.seed)

    starts = sample_unique_free(graph, args.num_agents, args.seed)
    tasks = generate_kiva_tasks(
        graph=graph,
        starts=starts,
        free_cells=[i for i in range(graph.num_vertices()) if not graph.is_blocked(i)],
        tasks_per_agent=args.tasks_per_agent,
        seed=args.seed,
        min_goal_distance=args.min_goal_distance,
    )

    stop_value = args.tasks_per_agent if args.stop_mode == "tasks" else args.ticks_limit

    if args.solver == "lacam":
        result = run_lacam(
            graph=graph,
            starts=starts,
            tasks=tasks,
            stop_mode=args.stop_mode,
            stop_value=stop_value,
            enable_clustering=not args.no_clustering,
            use_priority_open=args.priority_open,
            cluster_window=args.cluster_window,
        )
    else:
        result = run_pypibt(
            graph=graph,
            starts=starts,
            tasks=tasks,
            stop_mode=args.stop_mode,
            stop_value=stop_value,
            seed=args.seed,
        )

    total_completed = sum(result["completed_tasks"]) if result["completed_tasks"] else "n/a"
    print(
        f"[{args.solver}] ticks={result['ticks']}, total_moves={result['total_moves']}, "
        f"total_completed={total_completed}, runtime={result['runtime']:.2f}s"
    )
    if result.get("note"):
        print(f"note: {result['note']}")

    if _VIS_EXPORT_FLAG and result.get("path"):
        name = f"warehouse512_bulk_{args.solver}"
        map_path, sol_path = export_visualizer_files(name, graph, result["path"])
        vis_bin = shutil.which("mapf-visualizer-lifelong")
        if vis_bin and map_path and sol_path:
            try:
                subprocess.run([vis_bin, map_path, sol_path], check=True)
            except Exception as exc:
                print(f"⚠️  Визуализатор не запустился: {exc}")


if __name__ == "__main__":
    main()
