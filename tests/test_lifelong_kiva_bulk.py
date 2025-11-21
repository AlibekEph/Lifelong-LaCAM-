import os
import sys
import json
import argparse
import time
import shutil
import subprocess
import signal
from pathlib import Path
from collections import deque
from typing import List
import random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.graph.grid import GridGraph
from core.lifelong_lacam_integrated import LifelongLaCAMIntegrated
from core.configuration import Configuration
from strategies.generators.pibt import PIBTGenerator
from strategies.ordering.persistent_ordering import PersistentPriorityOrdering
from strategies.open_policy.stack import StackOpen
from strategies.open_policy.completed_priority import CompletedPriorityOpen
from utils.kiva_loader import layout_to_grid, generate_kiva_tasks

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


def run_lacam(
    graph: GridGraph,
    starts: list[int],
    tasks: list[list[int]],
    stop_mode: str,
    stop_value: int,
    enable_clustering: bool,
    cluster_window: int,
    use_priority_open: bool,
):
    tasks_runtime = [deque(agent_tasks) for agent_tasks in tasks]
    max_tasks_per_agent = stop_value if stop_mode == "tasks" else max(len(t) for t in tasks)
    initial_goals = [q[0] for q in tasks_runtime]

    def task_callback(agent_id: int, current_pos: int, old_goal: int) -> int:
        queue = tasks_runtime[agent_id]
        if queue and current_pos == queue[0]:
            queue.popleft()
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
        return {
            "ticks": max_iters,
            "total_moves": None,
            "completed_tasks": lacam.get_statistics().get("completed_tasks_per_agent"),
            "runtime": runtime,
            "note": "path not found before stop",
            "path": None,
        }
    ticks = len(path) - 1
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
    return {
        "ticks": ticks,
        "total_moves": total_mv,
        "completed_tasks": [len(tasks[aid]) - max(r, 0) for aid, r in enumerate(remaining)],
        "runtime": runtime,
        "note": None if (stop_mode != "tasks" or all(r <= 0 for r in remaining)) else "stopped by tick limit",
        "path": path,
    }


# ---------------- RHCR baseline ----------------


def _import_rhcr():
    """
    Lazy import for the RHCR solver from extern/gloriousDan-mapf.
    Adds local repo to sys.path and backfills typing.TypeAlias for Python <3.10.
    """
    import typing  # local import to avoid polluting global namespace

    mapf_root = Path(__file__).resolve().parents[1] / "extern" / "gloriousDan-mapf" / "mapf" / "src"
    if str(mapf_root) not in sys.path:
        sys.path.insert(0, str(mapf_root))

    if not hasattr(typing, "TypeAlias"):
        typing.TypeAlias = typing.Any  # type: ignore[attr-defined]

    try:
        from mapf.types.mapf_types import Vertex, MapfProblem, GridCell, GoalVerticesDict
        from mapf.solvers.common_functions import convert_gridworld_to_new_gridworld
        from mapf.solvers.rhcr_solver import RhcrSolver
        from mapf.types.mapf_config import MapfConfig, CbsCAT
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError(
            f"Не удалось импортировать RHCR из {mapf_root}. Установите зависимости (intervaltree) и проверьте репозиторий."
        ) from exc

    return Vertex, MapfProblem, GridCell, GoalVerticesDict, convert_gridworld_to_new_gridworld, RhcrSolver, MapfConfig, CbsCAT


def _import_cbs():
    """
    Lazy import for the CBS solver from extern/gloriousDan-mapf.
    Shares the same compatibility shims as RHCR.
    """
    import typing

    mapf_root = Path(__file__).resolve().parents[1] / "extern" / "gloriousDan-mapf" / "mapf" / "src"
    if str(mapf_root) not in sys.path:
        sys.path.insert(0, str(mapf_root))
    if not hasattr(typing, "TypeAlias"):
        typing.TypeAlias = typing.Any  # type: ignore[attr-defined]

    try:
        from mapf.types.mapf_types import Vertex, MapfProblem, GridCell, GoalVerticesDict
        from mapf.solvers.common_functions import convert_gridworld_to_new_gridworld
        from mapf.solvers.cbs_solver import CbsSolver
        from mapf.types.mapf_config import MapfConfig, CbsCAT
    except Exception as exc:
        raise ImportError(
            f"Не удалось импортировать CBS из {mapf_root}. Установите зависимости (intervaltree) и проверьте репозиторий."
        ) from exc

    return Vertex, MapfProblem, GridCell, GoalVerticesDict, convert_gridworld_to_new_gridworld, CbsSolver, MapfConfig, CbsCAT


def run_rhcr(
    graph: GridGraph,
    starts: list[int],
    tasks: list[list[int]],
    stop_mode: str,
    stop_value: int,
    timeout_s: int = 50,
):
    Vertex, MapfProblem, GridCell, GoalVerticesDict, convert_gridworld_to_new_gridworld, RhcrSolver, MapfConfig, CbsCAT = _import_rhcr()

    # convert grid to mapf format (x=col, y=row; True=blocked)
    grid_world = [[GridCell(bool(graph.grid[r, c])) for c in range(graph.W)] for r in range(graph.H)]
    new_grid = convert_gridworld_to_new_gridworld(grid_world)

    start_vertex = {aid: Vertex(graph.to_rc(pos)[1], graph.to_rc(pos)[0]) for aid, pos in enumerate(starts)}
    goal_vertices: GoalVerticesDict = {
        aid: tuple(Vertex(graph.to_rc(pos)[1], graph.to_rc(pos)[0]) for pos in agent_tasks)
        for aid, agent_tasks in enumerate(tasks)
    }

    config = MapfConfig()
    # slightly tighter defaults to reduce runtime; can be tweaked later
    config.RHCR_TIME_HORIZON_w = 5
    config.RHCR_REPLANNING_PERIOD_h = 5
    config.CBS_WINDOW = 5
    config.A_STAR_MAX_SEARCH_COUNT = 2000
    config.CBS_CONFLICT_AVOIDANCE = CbsCAT.ONLY_HIGHER

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
    signal.alarm(timeout_s)  # wall-clock guard
    t0 = time.perf_counter()
    try:
        solution, cost = RhcrSolver.solve_instance(problem, config)
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

    ticks = max(solution.keys())
    # compute total moves by comparing consecutive time steps
    moves = 0
    prev = None
    for t in sorted(solution.keys()):
        step = solution[t]
        if prev is not None:
            for aid, (v, _) in step.items():
                pv, _ = prev.get(aid, (v, 0))
                if pv != v:
                    moves += 1
        prev = step

    # count completed goals per agent
    completed = [0 for _ in tasks]
    progress = [0 for _ in tasks]
    for t in sorted(solution.keys()):
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


def run_cbs(
    graph: GridGraph,
    starts: list[int],
    tasks: list[list[int]],
    stop_mode: str,
    stop_value: int,
    timeout_s: int = 50,
):
    Vertex, MapfProblem, GridCell, GoalVerticesDict, convert_gridworld_to_new_gridworld, CbsSolver, MapfConfig, CbsCAT = _import_cbs()

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
    # limit CBS depth slightly via window if provided
    config.CBS_WINDOW = None

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
        solution, cost = CbsSolver.solve_instance(problem, config)
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

    ticks = max(solution.keys())
    moves = 0
    prev = None
    for t in sorted(solution.keys()):
        step = solution[t]
        if prev is not None:
            for aid, (v, _) in step.items():
                pv, _ = prev.get(aid, (v, 0))
                if pv != v:
                    moves += 1
        prev = step

    completed = [0 for _ in tasks]
    progress = [0 for _ in tasks]
    for t in sorted(solution.keys()):
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
    parser = argparse.ArgumentParser(description="Детерминированный тест с большим числом задач (Kiva)")
    parser.add_argument("--num-agents", type=int, default=100, help="Количество агентов")
    parser.add_argument("--tasks-per-agent", type=int, default=5000, help="Число задач на агента")
    parser.add_argument("--seed", type=int, default=0, help="Сид генерации стартов/задач")
    parser.add_argument("--solver", choices=["lacam", "pypibt", "rhcr", "cbs"], default="lacam", help="Выбор решателя")
    parser.add_argument("--stop-mode", choices=["tasks", "ticks"], default="tasks", help="Правило остановки")
    parser.add_argument("--ticks-limit", type=int, default=10000, help="Лимит тиков при stop-mode=ticks")
    parser.add_argument("--no-clustering", action="store_true", help="Отключить кластеризацию (для lacam)")
    parser.add_argument("--cluster-window", type=int, default=2, help="Максимальное окно кластеризации/PIBT (w)")
    parser.add_argument("--priority-open", action="store_true", help="Приоритет HL узлов по числу выполненных задач")
    parser.add_argument("--solver-timeout", type=int, default=50, help="Лимит по времени (сек) для внешних солверов (rhcr/cbs)")
    args = parser.parse_args()

    data_path = Path("data/kiva_large_tasks.json")
    assert data_path.exists(), "Сгенерируйте задачи через scripts/generate_kiva_tasks.py"
    payload = json.loads(data_path.read_text())
    grid = layout_to_grid(payload["layout"])
    graph = GridGraph(grid)

    # фиксируем RNG для воспроизводимости (PIBTGenerator использует random.shuffle)
    random.seed(args.seed)
    np.random.seed(args.seed)

    starts = sample_unique_free(graph, args.num_agents, args.seed)
    tasks = generate_kiva_tasks(
        graph=graph,
        starts=starts,
        free_cells=[i for i in range(graph.num_vertices()) if not graph.is_blocked(i)],
        tasks_per_agent=args.tasks_per_agent,
        seed=args.seed,
        min_goal_distance=4,
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
            cluster_window=args.cluster_window,
            use_priority_open=args.priority_open,
        )
    elif args.solver == "pypibt":
        result = run_pypibt(
            graph=graph,
            starts=starts,
            tasks=tasks,
            stop_mode=args.stop_mode,
            stop_value=stop_value,
            seed=args.seed,
        )
    elif args.solver == "rhcr":
        result = run_rhcr(
            graph=graph,
            starts=starts,
            tasks=tasks,
            stop_mode=args.stop_mode,
            stop_value=stop_value,
            timeout_s=args.solver_timeout,
        )
    else:
        result = run_cbs(
            graph=graph,
            starts=starts,
            tasks=tasks,
            stop_mode=args.stop_mode,
            stop_value=stop_value,
            timeout_s=args.solver_timeout,
        )

    print(
        f"[{args.solver}] ticks={result['ticks']}, total_moves={result['total_moves']}, "
        f"total_completed={sum(result['completed_tasks']) if result['completed_tasks'] else 'n/a'}, "
        f"runtime={result['runtime']:.2f}s"
    )
    if result.get("note"):
        print(f"note: {result['note']}")

    if _VIS_EXPORT_FLAG and result.get("path"):
        name = f"kiva_bulk_{args.solver}"
        map_path, sol_path = export_visualizer_files(name, graph, result["path"])
        vis_bin = shutil.which("mapf-visualizer-lifelong")
        if vis_bin and map_path and sol_path:
            try:
                subprocess.run([vis_bin, map_path, sol_path], check=True)
            except Exception as exc:
                print(f"⚠️  Визуализатор не запустился: {exc}")


if __name__ == "__main__":
    main()
