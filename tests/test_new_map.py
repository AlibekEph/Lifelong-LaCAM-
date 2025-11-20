import os
import sys
import json
import argparse
import time
import shutil
import subprocess
from pathlib import Path
from collections import deque
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.graph.grid import GridGraph
from core.lifelong_lacam_integrated import LifelongLaCAMIntegrated
from core.configuration import Configuration
from strategies.generators.pibt import PIBTGenerator
from strategies.ordering.persistent_ordering import PersistentPriorityOrdering
from strategies.open_policy.stack import StackOpen
from utils.kiva_loader import layout_to_grid, generate_kiva_tasks

# Новая карта (layout) для теста
NEW_MAP_LAYOUT = [
    ".......@.........@@.......@.....",
    ".....................@...@.....@",
    ".......@@......@..@@......@.....",
    "...............@.............@..",
    "@..............@........@.......",
    ".........@..........@.@.........",
    "@...@.@.........................",
    "........@..............@@.......",
    "..@...@.....@.@@...@......@.....",
    "........@...@.............@.....",
    ".................@..............",
    ".......@@....@......@..........@",
    "@....@.@...@.@.............@...@",
    "@........@.@..................@.",
    "....@..@@.......................",
    "........@......@....@...........",
    "@.................@............@",
    "............................@...",
    "@@....@........@.@..............",
    "...@......................@.@...",
    "....@...........................",
    "....@.........................@@",
    "@.............@.......@..@......",
    "....................@..@.@..@...",
    ".....@@..............@..........",
    "................................",
    ".@............@......@......@..@",
    ".......@........................",
    "...@.................@@........",
    ".........@............@.........",
    "....@..........@................",
    "...@...................@........",
]

def normalize_layout(rows: list[str]) -> list[str]:
    max_len = max(len(r) for r in rows)
    return [r.ljust(max_len, ".") for r in rows]


def export_visualizer_files(name: str, graph: GridGraph, config_list: list[Configuration]):
    os.makedirs("vis_outputs", exist_ok=True)
    map_path = os.path.join("vis_outputs", f"{name}.map")
    sol_path = os.path.join("vis_outputs", f"{name}.txt")

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
    import random

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


def run_lacam(graph: GridGraph, starts: list[int], tasks: list[list[int]], stop_mode: str, stop_value: int):
    tasks_runtime = [deque(agent_tasks) for agent_tasks in tasks]
    max_tasks_per_agent = stop_value if stop_mode == "tasks" else max(len(t) for t in tasks)
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
        max_tasks_per_agent=max_tasks_per_agent,
        enable_clustering=True,
    )

    t0 = time.time()
    max_iters = stop_value if stop_mode == "ticks" else 2_500_000
    path = lacam.run(max_iterations=max_iters, verbose=False)
    if path is None:
        try:
            node = lacam.open_policy.peek()  # type: ignore[attr-defined]
            path = node.reconstruct_path()
        except Exception:
            path = None
    runtime = time.time() - t0
    if path is None:
        return {"ticks": max_iters, "total_moves": None, "completed_tasks": lacam.get_statistics().get("completed_tasks_per_agent"), "runtime": runtime}
    ticks = len(path) - 1
    stats = lacam.get_statistics()
    return {
        "ticks": ticks,
        "total_moves": total_moves(path),
        "completed_tasks": stats.get("completed_tasks_per_agent"),
        "runtime": runtime,
        "path": path,
        "stats": stats,
    }


def run_pypibt(graph: GridGraph, starts: list[int], tasks: list[list[int]], stop_mode: str, stop_value: int, seed: int):
    try:
        from pypibt import PIBT
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "extern" / "pypibt" / "src"))
        import typing
        if not hasattr(typing, "TypeAlias"):
            from typing import Any
            typing.TypeAlias = Any  # type: ignore[attr-defined]
        from pypibt import PIBT  # type: ignore

    free_grid = (~graph.grid).astype(bool)
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

    max_iters = stop_value if stop_mode == "ticks" else 2_500_000
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
    stats = None
    return {
        "ticks": ticks,
        "total_moves": total_mv,
        "completed_tasks": [len(tasks[aid]) - max(r, 0) for aid, r in enumerate(remaining)],
        "runtime": runtime,
        "path": path,
        "stats": stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Тест на новой карте с двумя солверами")
    parser.add_argument("--solver", choices=["lacam", "pypibt"], default="pypibt")
    parser.add_argument("--stop-mode", choices=["tasks", "ticks"], default="ticks")
    parser.add_argument("--ticks-limit", type=int, default=50)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--tasks-per-agent", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    vis_enabled = os.environ.get("LACAM_VIS_EXPORT", "").lower() in {"", "1", "true", "yes", "on"}

    layout = normalize_layout(NEW_MAP_LAYOUT)
    grid = layout_to_grid(layout)
    graph = GridGraph(grid)
    starts = sample_unique_free(graph, args.num_agents, args.seed)
    tasks = generate_kiva_tasks(
        graph=graph,
        starts=starts,
        free_cells=[i for i in range(graph.num_vertices()) if not graph.is_blocked(i)],
        tasks_per_agent=args.tasks_per_agent,
        seed=args.seed,
        min_goal_distance=2,
    )

    stop_value = args.tasks_per_agent if args.stop_mode == "tasks" else args.ticks_limit

    if args.solver == "lacam":
        result = run_lacam(graph, starts, tasks, args.stop_mode, stop_value)
    else:
        result = run_pypibt(graph, starts, tasks, args.stop_mode, stop_value, args.seed)

    print(
        f"[{args.solver}] ticks={result['ticks']}, total_moves={result['total_moves']}, "
        f"total_completed={sum(result['completed_tasks']) if result.get('completed_tasks') else 'n/a'}, "
        f"runtime={result['runtime']:.2f}s"
    )
    if result.get("stats"):
        hl = (result["stats"] or {}).get("hl_metrics") or {}
        print(
            f"HL: runtime={hl.get('runtime_seconds', 0):.2f}s, "
            f"nodes={hl.get('hl_nodes_created', 0)}, "
            f"LL-exp={hl.get('ll_expansions', 0)}, "
            f"generator={{success:{hl.get('generator_successes', 0)}, fail:{hl.get('generator_failures', 0)}}}"
        )

    if vis_enabled and result.get("path"):
        name = f"new_map_{args.solver}"
        map_path, sol_path = export_visualizer_files(name, graph, result["path"])
        vis_bin = shutil.which("mapf-visualizer-lifelong")
        if vis_bin and map_path and sol_path:
            try:
                subprocess.run([vis_bin, map_path, sol_path], check=True)
            except Exception as exc:
                print(f"⚠️  Визуализатор не запустился: {exc}")


if __name__ == "__main__":
    main()
