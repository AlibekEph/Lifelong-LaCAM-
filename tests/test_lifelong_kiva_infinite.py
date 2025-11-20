import os
import sys
import json
import random
import shutil
import subprocess
import argparse
from pathlib import Path
from collections import deque
from typing import Deque, Iterable, List, Tuple
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.graph.grid import GridGraph
from core.lifelong_lacam_integrated import LifelongLaCAMIntegrated
from strategies.generators.pibt import PIBTGenerator
from strategies.ordering.persistent_ordering import PersistentPriorityOrdering
from strategies.open_policy.stack import StackOpen
from utils.kiva_loader import layout_to_grid
from core.configuration import Configuration

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


def _total_move_length(path: List) -> int:
    moves = 0
    for cur, nxt in zip(path, path[1:]):
        for u, v in zip(cur.pos, nxt.pos):
            if u != v:
                moves += 1
    return moves


def _sample_unique_positions(free_cells: List[int], k: int, rng: random.Random) -> List[int]:
    assert k <= len(free_cells), "Недостаточно свободных клеток для стартов"
    return rng.sample(free_cells, k)


def _task_stream(free_cells: List[int], rng: random.Random):
    while True:
        yield rng.choice(free_cells)


def _fill_queue(queue: Deque[int], stream: Iterable[int], target_len: int):
    it = iter(stream)
    while len(queue) < target_len:
        queue.append(next(it))

def _build_task_sources(num_agents: int, free_cells: List[int], tasks_per_agent: int, seed: int, buffer_len: int = 4):
    rng = random.Random(seed)
    streams = [_task_stream(free_cells, rng) for _ in range(num_agents)]
    tasks_runtime: List[Deque[int]] = [deque() for _ in range(num_agents)]
    tasks_for_episodes: List[Deque[int]] = [deque() for _ in range(num_agents)]
    remaining = [tasks_per_agent for _ in range(num_agents)]

    for aid in range(num_agents):
        target = min(buffer_len, remaining[aid])
        _fill_queue(tasks_runtime[aid], streams[aid], target)
        _fill_queue(tasks_for_episodes[aid], streams[aid], target)

    goals = [q[0] for q in tasks_runtime]
    return rng, streams, tasks_runtime, tasks_for_episodes, remaining, goals


def run_infinite_kiva(num_agents: int, tasks_per_agent: int, seed: int, enable_clustering: bool, quiet: bool):
    data_path = Path("data/kiva_large_tasks.json")
    assert data_path.exists(), "Сгенерируйте задачи через scripts/generate_kiva_tasks.py"
    payload = json.loads(data_path.read_text())

    layout = payload["layout"]
    grid = layout_to_grid(layout)
    graph = GridGraph(grid)

    free_cells = [idx for idx in range(graph.num_vertices()) if not graph.is_blocked(idx)]
    rng, task_streams, tasks_runtime, tasks_for_episodes, remaining, initial_goals = _build_task_sources(
        num_agents=num_agents,
        free_cells=free_cells,
        tasks_per_agent=tasks_per_agent,
        seed=seed,
    )
    starts = _sample_unique_positions(free_cells, num_agents, rng)

    def task_callback(agent_id: int, current_pos: int, old_goal: int) -> int:
        queue = tasks_runtime[agent_id]
        if queue and current_pos == queue[0]:
            queue.popleft()
            remaining[agent_id] -= 1
        if remaining[agent_id] <= 0:
            return old_goal
        target_len = min(4, remaining[agent_id])
        if len(queue) < target_len:
            _fill_queue(queue, task_streams[agent_id], target_len)
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

    start_time = time.time()
    path = lacam.run(max_iterations=2_000_000, verbose=False)
    assert path is not None, "План не найден"
    _validate_plan(graph, path, starts)

    ticks = len(path) - 1
    total_moves = _total_move_length(path)
    stats = lacam.get_statistics()
    runtime = time.time() - start_time

    if not quiet:
        print(f"[kiva_infinite] ticks={ticks}, total_moves={total_moves}, completed_tasks={stats['completed_tasks_per_agent']}, wall_time={runtime:.2f}s")
        hl = stats.get("hl_metrics") or {}
        print(
            f"HL: runtime={hl.get('runtime_seconds', 0):.2f}s, "
            f"nodes={hl.get('hl_nodes_created', 0)}, "
            f"LL-exp={hl.get('ll_expansions', 0)}, "
            f"generator={{success:{hl.get('generator_successes', 0)}, fail:{hl.get('generator_failures', 0)}}}"
        )

    if _VIS_EXPORT_FLAG:
        map_path, _ = export_visualizer_files("test_kiva_infinite", graph, path)
        def split_into_episodes(configs: list, queues: list[Deque[int]]) -> list[list]:
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
        ep_paths = []
        os.makedirs(_VIS_EXPORT_DIR, exist_ok=True)
        for idx, episode in enumerate(episodes, 1):
            ep_path = os.path.join(_VIS_EXPORT_DIR, f"test_kiva_infinite_ep{idx}.txt")
            with open(ep_path, "w", encoding="utf-8") as f:
                for t, conf in enumerate(episode):
                    coords = []
                    for pos in conf.pos:
                        r, c = graph.to_rc(pos)
                        coords.append(f"({c},{r})")
                    f.write(f"{t}:" + ",".join(coords) + ",\n")
            ep_paths.append(ep_path)
        vis_bin = shutil.which("mapf-visualizer-lifelong")
        if vis_bin and map_path and ep_paths:
            try:
                subprocess.run([vis_bin, map_path, *ep_paths], check=True)
            except Exception as exc:
                print(f"⚠️  Визуализатор не запустился: {exc}")

    return ticks, total_moves, stats


def run_infinite_kiva_pypibt(num_agents: int, tasks_per_agent: int, seed: int, quiet: bool):
    data_path = Path("data/kiva_large_tasks.json")
    assert data_path.exists(), "Сгенерируйте задачи через scripts/generate_kiva_tasks.py"
    payload = json.loads(data_path.read_text())

    layout = payload["layout"]
    grid = layout_to_grid(layout)
    graph = GridGraph(grid)

    free_cells = [idx for idx in range(graph.num_vertices()) if not graph.is_blocked(idx)]

    rng, task_streams, tasks_runtime, _, remaining, goals_idx = _build_task_sources(
        num_agents=num_agents,
        free_cells=free_cells,
        tasks_per_agent=tasks_per_agent,
        seed=seed,
    )
    starts = _sample_unique_positions(free_cells, num_agents, rng)

    positions_idx = list(starts)
    path = [Configuration(tuple(positions_idx))]
    ticks = 0
    total_moves = 0

    try:
        from pypibt import PIBT
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "extern" / "pypibt" / "src"))
        import typing
        if not hasattr(typing, "TypeAlias"):
            from typing import Any
            typing.TypeAlias = Any  # type: ignore[attr-defined]
        from pypibt import PIBT  # type: ignore

    free_grid = (~grid).astype(bool)  # pypibt ожидает True = свободно

    def idx_to_rc(idx: int) -> Tuple[int, int]:
        return graph.to_rc(idx)

    def rc_to_idx(rc: Tuple[int, int]) -> int:
        r, c = rc
        return graph.to_idx(r, c)

    max_iters = 2_000_000
    start_time = time.time()
    while ticks < max_iters:
        positions_rc = [idx_to_rc(p) for p in positions_idx]
        goals_rc = [idx_to_rc(g) for g in goals_idx]
        pibt = PIBT(free_grid, positions_rc, goals_rc, seed=seed + ticks)
        priorities = [pibt.dist_tables[i].get(positions_rc[i]) / free_grid.size for i in range(num_agents)]
        next_rc = pibt.step(positions_rc, priorities)
        next_idx = [rc_to_idx(rc) for rc in next_rc]

        total_moves += sum(1 for u, v in zip(positions_idx, next_idx) if u != v)
        ticks += 1
        positions_idx = next_idx
        path.append(Configuration(tuple(positions_idx)))

        for aid, pos in enumerate(positions_idx):
            if remaining[aid] <= 0 or not tasks_runtime[aid]:
                continue
            if pos == goals_idx[aid]:
                tasks_runtime[aid].popleft()
                remaining[aid] -= 1
                if remaining[aid] > 0:
                    target_len = min(4, remaining[aid])
                    if len(tasks_runtime[aid]) < target_len:
                        _fill_queue(tasks_runtime[aid], task_streams[aid], target_len)
                    goals_idx[aid] = tasks_runtime[aid][0]
        if all(r <= 0 for r in remaining):
            break

    assert all(r <= 0 for r in remaining), "pypibt не выполнил все задачи до лимита итераций"
    _validate_plan(graph, path, starts)
    runtime = time.time() - start_time
    if not quiet:
        print(f"[pypibt] ticks={ticks}, total_moves={total_moves}, completed_tasks={[tasks_per_agent - max(r, 0) for r in remaining]}, wall_time={runtime:.2f}s")
    return ticks, total_moves, {
        "completed_tasks_per_agent": [tasks_per_agent - max(r, 0) for r in remaining],
        "path": path,
        "runtime_seconds": runtime,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Бесконечный генератор задач для Lifelong LaCAM (Kiva map)")
    parser.add_argument("--num-agents", type=int, default=10, help="Количество агентов")
    parser.add_argument("--tasks-per-agent", type=int, default=100, help="Сколько задач должен выполнить каждый агент")
    parser.add_argument("--seed", type=int, default=0, help="Сид для генерации стартов/целей")
    parser.add_argument("--no-clustering", action="store_true", help="Отключить кластеризацию LL-шага")
    parser.add_argument("-q", "--quiet", action="store_true", help="Тихий режим")
    parser.add_argument("--solver", choices=["lacam", "pypibt"], default="lacam", help="Выбор решателя для сравнения")
    args = parser.parse_args()

    if args.solver == "lacam":
        run_infinite_kiva(
            num_agents=args.num_agents,
            tasks_per_agent=args.tasks_per_agent,
            seed=args.seed,
            enable_clustering=not args.no_clustering,
            quiet=args.quiet,
        )
    else:
        run_infinite_kiva_pypibt(
            num_agents=args.num_agents,
            tasks_per_agent=args.tasks_per_agent,
            seed=args.seed,
            quiet=args.quiet,
        )
