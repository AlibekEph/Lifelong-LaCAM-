import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from core.graph.grid import GridGraph
from core.lacam import LaCAM
from strategies.generators.pibt import PIBTGenerator
from strategies.open_policy.stack import StackOpen
from strategies.ordering.distance_ordering import DistanceOrdering


_VIS_EXPORT_ENV = os.environ.get("LACAM_VIS_EXPORT", "")
_VIS_EXPORT_FLAG = True if _VIS_EXPORT_ENV == "" else _VIS_EXPORT_ENV.lower() in {"1", "true", "yes", "on"}
_VIS_EXPORT_DIR = os.environ.get("LACAM_VIS_DIR", "vis_outputs")


def export_visualizer_files(name: str, graph: GridGraph, config_list: list):
    if not _VIS_EXPORT_FLAG:
        return
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


def validate_plan(graph: GridGraph, configs: list, starts: list[int], goals: list[int]):
    assert configs, "План пуст"
    assert configs[0].pos == tuple(starts), "Стартовая конфигурация неверна"
    assert configs[-1].pos == tuple(goals), "Финальная конфигурация неверна"

    for step, (cur, nxt) in enumerate(zip(configs, configs[1:]), 1):
        assert len(set(nxt.pos)) == len(nxt.pos), f"Vertex collision на шаге {step}"
        for aid, (u, v) in enumerate(zip(cur.pos, nxt.pos)):
            assert v == u or v in graph.neighbors(u), f"Агент {aid} делает недопустимый ход на шаге {step}"
            assert not graph.is_blocked(v), f"Агент {aid} вошёл в стену на шаге {step}"
        for i in range(len(cur.pos)):
            for j in range(i + 1, len(cur.pos)):
                if cur.pos[i] == nxt.pos[j] and cur.pos[j] == nxt.pos[i] and cur.pos[i] != cur.pos[j]:
                    raise AssertionError(f"Edge swap между агентами {i} и {j} на шаге {step}")


def test_simple_exchange():
    grid = np.zeros((5, 5), dtype=bool)
    graph = GridGraph(grid)
    starts = [graph.to_idx(0, 0), graph.to_idx(0, 4)]
    goals = [graph.to_idx(0, 4), graph.to_idx(0, 0)]

    lacam = LaCAM(
        graph=graph,
        starts=starts,
        goals=goals,
        generator=PIBTGenerator(),
        ordering=DistanceOrdering(),
        open_policy=StackOpen(),
        reinsert=True,
    )

    solution = lacam.run(max_iterations=50000)
    assert solution is not None
    validate_plan(graph, solution, starts, goals)
    export_visualizer_files("test1_simple_exchange", graph, solution)


def test_corridor_with_siding():
    grid = np.ones((3, 3), dtype=bool)
    grid[:, 1] = 0
    grid[1, 0] = 0
    graph = GridGraph(grid)
    starts = [graph.to_idx(0, 1), graph.to_idx(2, 1)]
    goals = [graph.to_idx(2, 1), graph.to_idx(0, 1)]

    lacam = LaCAM(
        graph=graph,
        starts=starts,
        goals=goals,
        generator=PIBTGenerator(),
        ordering=DistanceOrdering(),
        open_policy=StackOpen(),
        reinsert=False,
    )

    solution = lacam.run(max_iterations=5000)
    assert solution is not None
    validate_plan(graph, solution, starts, goals)
    export_visualizer_files("test4_corridor_with_siding", graph, solution)


def test_with_obstacles():
    grid = np.zeros((9, 9), dtype=bool)
    grid[2:5, 3:6] = True
    graph = GridGraph(grid)
    starts = [graph.to_idx(0, 0), graph.to_idx(8, 8)]
    goals = [graph.to_idx(8, 8), graph.to_idx(0, 0)]

    lacam = LaCAM(
        graph=graph,
        starts=starts,
        goals=goals,
        generator=PIBTGenerator(),
        ordering=DistanceOrdering(),
        open_policy=StackOpen(),
        reinsert=True,
    )

    solution = lacam.run(max_iterations=50000)
    assert solution is not None
    validate_plan(graph, solution, starts, goals)
    export_visualizer_files("test3_with_obstacles", graph, solution)


def test_kishka():
    grid = np.array([[0, 0, 0], [1, 0, 1]], dtype=bool)
    graph = GridGraph(grid)
    starts = [graph.to_idx(0, 0), graph.to_idx(0, 2)]
    goals = [graph.to_idx(0, 2), graph.to_idx(0, 0)]

    lacam = LaCAM(
        graph=graph,
        starts=starts,
        goals=goals,
        generator=PIBTGenerator(),
        ordering=DistanceOrdering(),
        open_policy=StackOpen(),
        reinsert=True,
    )
def test_perek():
    grid = np.array([[1, 1, 0, 1, 1], 
                     [1, 1, 0, 1, 1],
                     [0, 0, 0, 0, 0],
                     [1, 1, 0, 1, 1],
                     [1, 1, 0, 1, 1]], dtype=bool)
    graph = GridGraph(grid)
    starts = [graph.to_idx(2, 0), 
              graph.to_idx(0, 2), 
              graph.to_idx(4, 2),
              graph.to_idx(2, 4),
              ]
    goals = [graph.to_idx(4, 2), 
              graph.to_idx(2, 4), 
              graph.to_idx(0, 2),
              graph.to_idx(2, 0),
              ]

    lacam = LaCAM(
        graph=graph,
        starts=starts,
        goals=goals,
        generator=PIBTGenerator(),
        ordering=DistanceOrdering(),
        open_policy=StackOpen(),
        reinsert=True,
    )

    solution = lacam.run(max_iterations=50000)
    assert solution is not None
    validate_plan(graph, solution, starts, goals)
    export_visualizer_files("test5_perek", graph, solution)

if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__]))
