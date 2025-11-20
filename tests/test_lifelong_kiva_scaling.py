import os
import sys
import json
import time
from pathlib import Path
from collections import deque

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.graph.grid import GridGraph
from core.lifelong_lacam_integrated import LifelongLaCAMIntegrated
from strategies.generators.pibt import PIBTGenerator
from strategies.ordering.distance_ordering import DistanceOrdering
from strategies.open_policy.stack import StackOpen
from utils.kiva_loader import layout_to_grid, coords_to_indices


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


def _pretty_print_metrics(stats: dict, label: str):
    hl = stats.get("hl_metrics") or {}
    gen = stats.get("generator_metrics") or {}
    stack = gen.get("stack_depth") or {}
    runtime = hl.get("runtime_seconds")
    runtime_str = f"{runtime:.2f}s" if runtime is not None else "n/a"
    print(f"[{label}] HL metrics:")
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
    print(f"[{label}] PIBT metrics:")
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
            f"max={stack.get('max', 0)}, p50={stack.get('p50', 0)}, "
            f"p90={stack.get('p90', 0)}, p99={stack.get('p99', 0)}"
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


TASK_LEVELS = [120, 140, 150, 160, 170, 200, 230, 250, 300, 500, 1000]


@pytest.mark.parametrize("tasks_per_agent", TASK_LEVELS)
def test_lifelong_kiva_scaled(tasks_per_agent: int):
    data_path = Path("data/kiva_large_tasks.json")
    assert data_path.exists(), "Сгенерируйте задачи через scripts/generate_kiva_tasks.py"
    payload = json.loads(data_path.read_text())

    layout = payload["layout"]
    grid = layout_to_grid(layout)
    graph = GridGraph(grid)

    agents_to_use = min(15, len(payload["starts"]))
    starts_coords = payload["starts"][:agents_to_use]
    tasks_coords_list = payload["tasks"][:agents_to_use]
    starts = coords_to_indices(graph, starts_coords)
    base_tasks_indices = [coords_to_indices(graph, coords) for coords in tasks_coords_list]

    tasks_list = [_repeat_to_length(agent_tasks, tasks_per_agent) for agent_tasks in base_tasks_indices]
    tasks_runtime = [deque(agent_tasks) for agent_tasks in tasks_list]
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

    start = time.time()
    max_iters = max(2_000_000, tasks_per_agent * 500)
    path = lacam.run(max_iterations=max_iters, verbose=False)
    duration = time.time() - start
    assert path is not None, f"План не найден при {tasks_per_agent} задачах"

    _validate_plan(graph, path, starts)
    stats = lacam.get_statistics()
    print(f"[kiva_{tasks_per_agent}] solved in {duration:.2f}s")
    assert all(x >= tasks_per_agent for x in stats["completed_tasks_per_agent"])
    _pretty_print_metrics(stats, label=f"kiva_{tasks_per_agent}")
