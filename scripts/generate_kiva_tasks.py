#!/usr/bin/env python3
"""
Generate task queues for Lifelong LaCAM using a Kiva-style warehouse map.

Example:
    python scripts/generate_kiva_tasks.py \
        --map-json ggo_public/maps/warehouse/expert_baseline/kiva_large_w_mode_flow_baseline.json \
        --num-agents 50 \
        --tasks-per-agent 100 \
        --seed 42 \
        --output data/kiva_large_tasks.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np  # noqa: F401  (ensures numpy available before GridGraph)

from core.graph.grid import GridGraph
from utils.kiva_loader import (
    KivaMap,
    generate_kiva_tasks,
    load_kiva_map,
    indices_to_coords,
)


def sample_agent_starts(kiva: KivaMap, num_agents: int, seed: int) -> List[int]:
    if num_agents > len(kiva.free_cells):
        raise ValueError("Number of agents exceeds available free cells")
    rng = np.random.default_rng(seed)
    choices = rng.choice(kiva.free_cells, size=num_agents, replace=False)
    return [int(v) for v in choices]


def main():
    parser = argparse.ArgumentParser(description="Generate Lifelong tasks for a Kiva map.")
    parser.add_argument("--map-json", required=True, help="Path to kiva_* JSON layout")
    parser.add_argument("--num-agents", type=int, required=True, help="Number of agents")
    parser.add_argument("--tasks-per-agent", type=int, default=100, help="Tasks per agent")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()

    kiva = load_kiva_map(args.map_json)
    starts_idx = sample_agent_starts(kiva, args.num_agents, args.seed)
    tasks_idx = generate_kiva_tasks(
        graph=kiva.graph,
        starts=starts_idx,
        free_cells=kiva.free_cells,
        tasks_per_agent=args.tasks_per_agent,
        seed=args.seed,
    )

    starts_coords = indices_to_coords(kiva.graph, starts_idx)
    tasks_coords = [indices_to_coords(kiva.graph, agent_tasks) for agent_tasks in tasks_idx]

    out_data = {
        "map": Path(args.map_json).name,
        "layout": kiva.layout,
        "num_agents": args.num_agents,
        "tasks_per_agent": args.tasks_per_agent,
        "starts": starts_coords,
        "tasks": tasks_coords,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out_data, indent=2))
    print(f"Saved tasks to {output_path}")


if __name__ == "__main__":
    main()
