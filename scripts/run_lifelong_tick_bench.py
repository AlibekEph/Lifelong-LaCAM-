"""
Batch runner for tick-limited lifelong MAPF benchmarks.

- Varies number of agents (1..20)
- Varies tick limits in a fixed list (10..1000)
- Runs lacam, pypibt, rhcr, cbs
- Stops running a solver for a given agent count after the first timeout
- Saves per-agent plots (tasks completed vs ticks) to output directory
"""

from __future__ import annotations

import subprocess
import sys
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# dynamic import of test utilities
ticks_path = ROOT / "tests" / "test_lifelong_kiva_ticks.py"
spec = importlib.util.spec_from_file_location("kiva_ticks", ticks_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Не удалось загрузить модуль {ticks_path}")
kiva_ticks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kiva_ticks)

from core.graph.grid import GridGraph
from utils.kiva_loader import layout_to_grid, generate_kiva_tasks


# Configuration
AGENT_COUNTS = list(range(1, 21))
TICK_VALUES = [10, 50, 100, 200, 500, 1000]
SOLVERS = ["lacam", "pypibt", "rhcr", "cbs"]
SEED = 0
SOLVER_TIMEOUT = 15  # seconds for rhcr/cbs
TASKS_PER_AGENT_OFFLINE = max(TICK_VALUES) * 2  # чтобы offline решатели не кончали задачи
OUTPUT_DIR = Path("bench_plots")


def load_graph() -> GridGraph:
    data_path = Path("data/kiva_large_tasks.json")
    payload = json.loads(data_path.read_text())
    grid = layout_to_grid(payload["layout"])
    return GridGraph(grid)


def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_solver(
    solver: str,
    graph: GridGraph,
    starts: List[int],
    tick_limit: int,
    seed: int,
    offline_tasks: List[List[int]] | None,
) -> Dict:
    if solver == "lacam":
        return kiva_ticks.run_lacam_ticks(graph=graph, starts=starts, ticks_limit=tick_limit, seed=seed, use_priority_open=False)
    if solver == "pypibt":
        return kiva_ticks.run_pypibt_ticks(graph=graph, starts=starts, ticks_limit=tick_limit, seed=seed)
    # offline solvers use pre-generated tasks
    assert offline_tasks is not None
    return kiva_ticks._run_offline_solver(
        solver_name=solver,
        graph=graph,
        starts=starts,
        tasks=offline_tasks,
        ticks_limit=tick_limit,
        timeout_s=SOLVER_TIMEOUT,
    )


def make_offline_tasks(graph: GridGraph, starts: List[int], seed: int) -> List[List[int]]:
    free_cells = [i for i in range(graph.num_vertices()) if not graph.is_blocked(i)]
    return generate_kiva_tasks(
        graph=graph,
        starts=starts,
        free_cells=free_cells,
        tasks_per_agent=TASKS_PER_AGENT_OFFLINE,
        seed=seed,
        min_goal_distance=4,
    )


def main():
    ensure_output_dir()
    graph = load_graph()

    for num_agents in AGENT_COUNTS:
        print(f"\n=== Agents: {num_agents} ===")
        starts = kiva_ticks.sample_unique_free(graph, num_agents, SEED)
        offline_tasks = make_offline_tasks(graph, starts, SEED)

        results: Dict[str, List[Tuple[int, int]]] = {solver: [] for solver in SOLVERS}
        timeouts: Dict[str, bool] = {solver: False for solver in SOLVERS}

        for tick_limit in TICK_VALUES:
            for solver in SOLVERS:
                if timeouts[solver]:
                    continue
                res = run_solver(
                    solver=solver,
                    graph=graph,
                    starts=starts,
                    tick_limit=tick_limit,
                    seed=SEED,
                    offline_tasks=offline_tasks if solver in {"rhcr", "cbs"} else None,
                )
                tasks_done = sum(res["completed_tasks"]) if res.get("completed_tasks") else None
                results[solver].append((tick_limit, tasks_done if tasks_done is not None else 0))
                print(
                    f"{solver}@{tick_limit} ticks -> tasks_done={tasks_done}, runtime={res.get('runtime'):.2f}s, note={res.get('note')}"
                )
                note = res.get("note") or ""
                if tasks_done is None or note.startswith("timeout"):
                    timeouts[solver] = True

        # build plot for this agent count
        plt.figure()
        for solver in SOLVERS:
            if not results[solver]:
                continue
            xs = [t for t, _ in results[solver]]
            ys = [v for _, v in results[solver]]
            plt.plot(xs, ys, marker="o", label=solver)
        plt.xlabel("Ticks")
        plt.ylabel("Completed tasks")
        plt.title(f"Agents={num_agents}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        out_path = OUTPUT_DIR / f"agents_{num_agents}.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
