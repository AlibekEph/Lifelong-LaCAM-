"""Lifelong MAPF tick-limited benchmark.

Запускает решатели с бесконечной подачей задач, ограничивает симуляцию числом тиков.
Метрика: сколько задач завершено за заданное число тиков.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import time
from collections import deque
from pathlib import Path
from typing import List

import numpy as np

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.configuration import Configuration
from core.graph.grid import GridGraph
from core.lifelong_lacam_integrated import LifelongLaCAMIntegrated
from strategies.generators.pibt import PIBTGenerator
from strategies.open_policy.completed_priority import CompletedPriorityOpen
from strategies.open_policy.stack import StackOpen
from strategies.ordering.persistent_ordering import PersistentPriorityOrdering
from utils.kiva_loader import generate_kiva_tasks, layout_to_grid


def sample_unique_free(graph: GridGraph, num: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    free = [idx for idx in range(graph.num_vertices()) if not graph.is_blocked(idx)]
    assert len(free) >= num, "Недостаточно свободных клеток для стартов"
    return rng.sample(free, num)


def make_task_sampler(graph: GridGraph, free_cells: list[int], min_goal_distance: int, seed: int):
    rng = np.random.default_rng(seed)

    def sample_goal(current: int) -> int:
        """Подбор новой цели на лету с проверкой достижимости и минимального расстояния."""
        attempt_cap = len(free_cells) * 10
        for _ in range(attempt_cap):
            goal = int(rng.choice(free_cells))
            if goal != current and graph.dist(current, goal) >= min_goal_distance:
                return goal
        # fallback: ближайшая достижимая
        reachable = [cell for cell in free_cells if graph.dist(current, cell) >= 0 and cell != current]
        if not reachable:
            raise RuntimeError("Нет достижимых вершин для новых задач")
        return min(reachable, key=lambda cell: graph.dist(current, cell))

    return sample_goal


def total_moves(path: list[Configuration]) -> int:
    moves = 0
    for cur, nxt in zip(path, path[1:]):
        for u, v in zip(cur.pos, nxt.pos):
            if u != v:
                moves += 1
    return moves


def run_lacam_ticks(graph: GridGraph, starts: list[int], ticks_limit: int, seed: int, use_priority_open: bool):
    free_cells = [i for i in range(graph.num_vertices()) if not graph.is_blocked(i)]
    sample_goal = make_task_sampler(graph, free_cells, min_goal_distance=4, seed=seed)

    tasks_runtime = [deque([sample_goal(pos)]) for pos in starts]
    completed = [0 for _ in starts]
    initial_goals = [q[0] for q in tasks_runtime]

    def task_callback(agent_id: int, current_pos: int, old_goal: int) -> int:
        queue = tasks_runtime[agent_id]
        if queue and current_pos == queue[0]:
            queue.popleft()
            completed[agent_id] += 1
            queue.append(sample_goal(current_pos))
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
        max_tasks_per_agent=10_000_000,  # effectively infinite
        enable_clustering=True,
        cluster_window_w=2,
    )

    t0 = time.perf_counter()
    path = lacam.run(max_iterations=ticks_limit, verbose=False)
    runtime = time.perf_counter() - t0
    if path is None:
        try:
            node = lacam.open_policy.peek()  # type: ignore[attr-defined]
            path = node.reconstruct_path()
        except Exception:
            path = None
    ticks = len(path) - 1 if path else ticks_limit
    return {
        "ticks": ticks,
        "total_moves": total_moves(path) if path else None,
        "completed_tasks": completed,
        "runtime": runtime,
        "note": None if path else "path not found before stop",
        "path": path,
    }


def run_pypibt_ticks(graph: GridGraph, starts: list[int], ticks_limit: int, seed: int):
    try:
        from pypibt import PIBT
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "extern" / "pypibt" / "src"))
        import typing
        if not hasattr(typing, "TypeAlias"):
            from typing import Any
            typing.TypeAlias = Any  # type: ignore[attr-defined]
        from pypibt import PIBT  # type: ignore

    free_cells = [i for i in range(graph.num_vertices()) if not graph.is_blocked(i)]
    sample_goal = make_task_sampler(graph, free_cells, min_goal_distance=4, seed=seed)

    free_grid = (~graph.grid).astype(bool)
    positions = list(starts)
    goals = [sample_goal(pos) for pos in positions]
    tasks_runtime = [deque([g]) for g in goals]
    completed = [0 for _ in starts]
    path = [Configuration(tuple(positions))]
    ticks = 0
    total_mv = 0
    t0 = time.perf_counter()

    def idx_to_rc(idx: int) -> tuple[int, int]:
        return graph.to_rc(idx)

    def rc_to_idx(rc: tuple[int, int]) -> int:
        r, c = rc
        return graph.to_idx(r, c)

    while ticks < ticks_limit:
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
            if pos == goals[aid]:
                completed[aid] += 1
                new_goal = sample_goal(pos)
                tasks_runtime[aid].append(new_goal)
                goals[aid] = new_goal

    runtime = time.perf_counter() - t0
    return {
        "ticks": ticks,
        "total_moves": total_mv,
        "completed_tasks": completed,
        "runtime": runtime,
        "note": None,
        "path": path,
    }


def _import_glorious_mapf():
    import typing

    mapf_root = Path(__file__).resolve().parents[1] / "extern" / "gloriousDan-mapf" / "mapf" / "src"
    if str(mapf_root) not in sys.path:
        sys.path.insert(0, str(mapf_root))
    if not hasattr(typing, "TypeAlias"):
        typing.TypeAlias = typing.Any  # type: ignore[attr-defined]

    from mapf.types.mapf_types import Vertex, MapfProblem, GridCell, GoalVerticesDict
    from mapf.solvers.common_functions import convert_gridworld_to_new_gridworld
    from mapf.solvers.rhcr_solver import RhcrSolver
    from mapf.solvers.cbs_solver import CbsSolver
    from mapf.types.mapf_config import MapfConfig, CbsCAT

    return Vertex, MapfProblem, GridCell, GoalVerticesDict, convert_gridworld_to_new_gridworld, RhcrSolver, CbsSolver, MapfConfig, CbsCAT


def _run_offline_solver(
    solver_name: str,
    graph: GridGraph,
    starts: list[int],
    tasks: list[list[int]],
    ticks_limit: int,
    timeout_s: int,
):
    (
        Vertex,
        MapfProblem,
        GridCell,
        GoalVerticesDict,
        convert_gridworld_to_new_gridworld,
        RhcrSolver,
        CbsSolver,
        MapfConfig,
        CbsCAT,
    ) = _import_glorious_mapf()

    grid_world = [[GridCell(bool(graph.grid[r, c])) for c in range(graph.W)] for r in range(graph.H)]
    new_grid = convert_gridworld_to_new_gridworld(grid_world)

    start_vertex = {aid: Vertex(graph.to_rc(pos)[1], graph.to_rc(pos)[0]) for aid, pos in enumerate(starts)}
    goal_vertices: GoalVerticesDict = {
        aid: tuple(Vertex(graph.to_rc(pos)[1], graph.to_rc(pos)[0]) for pos in agent_tasks)
        for aid, agent_tasks in enumerate(tasks)
    }

    config = MapfConfig()
    config.CBS_CONFLICT_AVOIDANCE = CbsCAT.ONLY_HIGHER
    config.A_STAR_MAX_SEARCH_COUNT = 2000
    if solver_name == "rhcr":
        config.RHCR_TIME_HORIZON_w = 5
        config.RHCR_REPLANNING_PERIOD_h = 5
        config.CBS_WINDOW = 5

    problem = MapfProblem(
        agent_ids=list(range(len(starts))),
        start_vertex=start_vertex,
        grid=new_grid,
        init_goal_vertices=goal_vertices,
    )

    class _Timeout(Exception):
        pass

    def _handler(_signum, _frame):
        raise _Timeout()

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_s)
    t0 = time.perf_counter()
    try:
        if solver_name == "rhcr":
            solution, _ = RhcrSolver.solve_instance(problem, config)
        else:
            solution, _ = CbsSolver.solve_instance(problem, config)
        runtime = time.perf_counter() - t0
    except _Timeout:
        runtime = time.perf_counter() - t0
        return {
            "ticks": None,
            "total_moves": None,
            "completed_tasks": None,
            "runtime": runtime,
            "note": f"timeout ({timeout_s}s wall clock)",
            "path": None,
        }
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    if not solution:
        return {
            "ticks": None,
            "total_moves": None,
            "completed_tasks": None,
            "runtime": runtime,
            "note": "no solution",
            "path": None,
        }

    ticks = min(ticks_limit, max(solution.keys()))
    moves = 0
    prev = None
    for t in sorted(k for k in solution.keys() if k <= ticks_limit):
        step = solution[t]
        if prev is not None:
            for aid, (v, _) in step.items():
                pv, _ = prev.get(aid, (v, 0))
                if pv != v:
                    moves += 1
        prev = step

    completed = [0 for _ in tasks]
    progress = [0 for _ in tasks]
    for t in sorted(k for k in solution.keys() if k <= ticks_limit):
        step = solution[t]
        for aid, goals in enumerate(tasks):
            if progress[aid] >= len(goals):
                continue
            v_goal = goal_vertices[aid][progress[aid]]
            v_cur, _ = step.get(aid, (None, 0))
            if v_cur == v_goal:
                progress[aid] += 1
                completed[aid] += 1

    return {
        "ticks": ticks,
        "total_moves": moves,
        "completed_tasks": completed,
        "runtime": runtime,
        "note": None,
        "path": None,
    }


def main():
    parser = argparse.ArgumentParser(description="Tick-limited lifelong MAPF benchmark with infinite tasks.")
    parser.add_argument("--num-agents", type=int, default=10, help="Количество агентов")
    parser.add_argument("--ticks-limit", type=int, default=1000, help="Лимит тиков симуляции")
    parser.add_argument("--seed", type=int, default=0, help="Сид генерации")
    parser.add_argument("--solver", choices=["lacam", "pypibt", "rhcr", "cbs"], default="lacam", help="Выбор решателя")
    parser.add_argument("--priority-open", action="store_true", help="Для lacam: приоритет HL по числу выполненных задач")
    parser.add_argument("--solver-timeout", type=int, default=50, help="Таймаут для RHCR/CBS")
    parser.add_argument("--tasks-per-agent-offline", type=int, default=200, help="Число задач для оффлайн солверов (rhcr/cbs)")
    args = parser.parse_args()

    data_path = Path("data/kiva_large_tasks.json")
    assert data_path.exists(), "Сгенерируйте задачи через scripts/generate_kiva_tasks.py"
    payload = json.loads(data_path.read_text())
    grid = layout_to_grid(payload["layout"])
    graph = GridGraph(grid)

    random.seed(args.seed)
    np.random.seed(args.seed)

    starts = sample_unique_free(graph, args.num_agents, args.seed)
    free_cells = [i for i in range(graph.num_vertices()) if not graph.is_blocked(i)]

    if args.solver in {"rhcr", "cbs"}:
        # для оффлайн решателей заранее генерируем длинные последовательности
        tasks = generate_kiva_tasks(
            graph=graph,
            starts=starts,
            free_cells=free_cells,
            tasks_per_agent=args.tasks_per_agent_offline,
            seed=args.seed,
            min_goal_distance=4,
        )
        result = _run_offline_solver(
            solver_name=args.solver,
            graph=graph,
            starts=starts,
            tasks=tasks,
            ticks_limit=args.ticks_limit,
            timeout_s=args.solver_timeout,
        )
    elif args.solver == "pypibt":
        result = run_pypibt_ticks(
            graph=graph,
            starts=starts,
            ticks_limit=args.ticks_limit,
            seed=args.seed,
        )
    else:
        result = run_lacam_ticks(
            graph=graph,
            starts=starts,
            ticks_limit=args.ticks_limit,
            seed=args.seed,
            use_priority_open=args.priority_open,
        )

    print(
        f"[{args.solver}] ticks={result['ticks']}, total_moves={result['total_moves']}, "
        f"tasks_done={sum(result['completed_tasks']) if result['completed_tasks'] else 'n/a'}, "
        f"runtime={result['runtime']:.2f}s"
    )
    if result.get("note"):
        print(f"note: {result['note']}")


if __name__ == "__main__":
    main()
