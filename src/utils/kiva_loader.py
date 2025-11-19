from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

from core.graph.grid import GridGraph


KIVA_BLOCKED_CHARS = {"@", "#"}


def layout_to_grid(layout: Sequence[str]) -> np.ndarray:
    """
    Convert a layout from GGO/G2O format into boolean numpy grid.

    Characters interpreted as:
        '@' or '#'  – shelves/blocked
        everything else ('.', 'w', 'e', etc.) – walkable
    """
    if not layout:
        raise ValueError("Layout is empty")
    width = len(layout[0])
    for row in layout:
        if len(row) != width:
            raise ValueError("Layout rows have inconsistent widths")

    grid = np.zeros((len(layout), width), dtype=bool)
    for r, row in enumerate(layout):
        for c, ch in enumerate(row):
            grid[r, c] = ch in KIVA_BLOCKED_CHARS
    return grid


@dataclass
class KivaMap:
    graph: GridGraph
    layout: List[str]
    free_cells: List[int]


def load_kiva_map(json_path: str | Path) -> KivaMap:
    """Load a Kiva-style warehouse map from the provided JSON."""
    data = json.loads(Path(json_path).read_text())
    layout = data.get("layout")
    if not layout or not isinstance(layout, list):
        raise ValueError(f"JSON {json_path} does not contain 'layout' list")

    grid = layout_to_grid(layout)
    graph = GridGraph(grid)
    free_cells = [
        graph.to_idx(r, c)
        for r in range(graph.H)
        for c in range(graph.W)
        if not grid[r, c]
    ]
    return KivaMap(graph=graph, layout=layout, free_cells=free_cells)


def ensure_reachable(graph: GridGraph, u: int, v: int) -> bool:
    """Check reachability via GridGraph BFS distance."""
    return graph.dist(u, v) >= 0


def generate_kiva_tasks(
    graph: GridGraph,
    starts: List[int],
    free_cells: Sequence[int],
    tasks_per_agent: int,
    *,
    seed: int = 0,
    min_goal_distance: int = 4,
) -> List[List[int]]:
    """
    Generate task sequences for each agent by sampling random reachable targets.

    Args:
        graph: warehouse GridGraph
        starts: starting positions for agents (indices in GridGraph)
        free_cells: list of walkable vertices to sample from
        tasks_per_agent: number of tasks per agent
        seed: RNG seed for reproducibility
        min_goal_distance: reject goals closer than this to current position
    """
    if tasks_per_agent <= 0:
        raise ValueError("tasks_per_agent must be > 0")
    if not free_cells:
        raise ValueError("free_cells list is empty")

    rng = np.random.default_rng(seed)
    free_cells = list(free_cells)
    tasks: List[List[int]] = [[] for _ in starts]

    for aid, start in enumerate(starts):
        current = start
        agent_tasks: List[int] = []
        attempt_cap = len(free_cells) * 10

        for _ in range(tasks_per_agent):
            tries = 0
            goal = current
            while (goal == current or graph.dist(current, goal) < min_goal_distance) and tries < attempt_cap:
                goal = int(rng.choice(free_cells))
                tries += 1
            if not ensure_reachable(graph, current, goal):
                # fallback: choose the closest reachable free cell
                reachable = [
                    cell for cell in free_cells if ensure_reachable(graph, current, cell)
                ]
                if not reachable:
                    raise RuntimeError(f"No reachable vertices for agent {aid}")
                goal = min(reachable, key=lambda cell: graph.dist(current, cell))
            agent_tasks.append(goal)
            current = goal
        tasks[aid] = agent_tasks
    return tasks


def indices_to_coords(graph: GridGraph, vertices: Sequence[int]) -> List[tuple[int, int]]:
    """Convert list of vertex indices to (row, col) coordinates."""
    return [graph.to_rc(v) for v in vertices]


def coords_to_indices(graph: GridGraph, coords: Sequence[Sequence[int]]) -> List[int]:
    """Convert iterable of (row, col) to vertex indices."""
    result = []
    for rc in coords:
        if len(rc) != 2:
            raise ValueError(f"Coordinate must have length 2, got {rc}")
        r, c = rc
        result.append(graph.to_idx(int(r), int(c)))
    return result
