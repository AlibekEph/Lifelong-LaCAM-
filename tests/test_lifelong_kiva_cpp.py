import os
import sys
import json
import argparse
import time
import subprocess
import tempfile
from pathlib import Path
import random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.graph.grid import GridGraph
from utils.kiva_loader import layout_to_grid, generate_kiva_tasks

def sample_unique_free(graph: GridGraph, num: int, seed: int) -> list[int]:
    import random
    rng = random.Random(seed)
    free = [idx for idx in range(graph.num_vertices()) if not graph.is_blocked(idx)]
    assert len(free) >= num, "Not enough free cells for starts"
    return rng.sample(free, num)

def run_lacam_cpp(
    graph: GridGraph,
    starts: list[int],
    tasks: list[list[int]],
    num_agents: int,
    tasks_per_agent: int,
    layout_str: list[str],
):
    # Prepare data for JSON
    # C++ expects starts as [row, col] and tasks as [[row, col], ...]
    starts_rc = [list(graph.to_rc(s)) for s in starts]
    tasks_rc = []
    for agent_tasks in tasks:
        agent_tasks_rc = [list(graph.to_rc(t)) for t in agent_tasks]
        tasks_rc.append(agent_tasks_rc)

    data = {
        "layout": layout_str,
        "starts": starts_rc,
        "tasks": tasks_rc
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        json.dump(data, tmp)
        tmp_path = tmp.name

    try:
        # Run C++ binary
        # Usage: ./lifelong_lacam_cpp <json_path> <num_agents> <tasks_per_agent>
        binary_path = os.path.join(os.path.dirname(__file__), "..", "src_cpp", "build", "lifelong_lacam_cpp")
        cmd = [binary_path, tmp_path, str(num_agents), str(tasks_per_agent)]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout
        print("STDERR:", result.stderr)
        
        # Parse output
        path_length = 0
        runtime = 0.0
        total_moves = 0
        completed_tasks = []
        
        for line in output.splitlines():
            if line.startswith("Path length:"):
                path_length = int(line.split(":")[1].strip())
            elif line.startswith("Runtime:"):
                runtime = float(line.split(":")[1].strip().replace("s", ""))
            elif line.startswith("Total moves:"):
                total_moves = int(line.split(":")[1].strip())
            elif line.startswith("Completed tasks:"):
                parts = line.split(":")[1].strip().split()
                if parts:
                    completed_tasks = [int(x) for x in parts]
                else:
                    completed_tasks = [0] * num_agents

        return {
            "ticks": path_length,
            "total_moves": total_moves,
            "completed_tasks": completed_tasks,
            "runtime": runtime,
            "note": None
        }

    finally:
        # if os.path.exists(tmp_path):
        #     os.remove(tmp_path)
        print(f"Temp JSON: {tmp_path}")

def main():
    parser = argparse.ArgumentParser(description="Bulk test for C++ Lifelong LaCAM")
    parser.add_argument("--num-agents", type=int, default=100, help="Number of agents")
    parser.add_argument("--tasks-per-agent", type=int, default=5000, help="Tasks per agent")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    args = parser.parse_args()

    data_path = Path("data/kiva_large_tasks.json")
    assert data_path.exists(), "Generate tasks via scripts/generate_kiva_tasks.py"
    payload = json.loads(data_path.read_text())
    layout = payload["layout"]
    grid = layout_to_grid(layout)
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
        min_goal_distance=4,
    )

    result = run_lacam_cpp(
        graph=graph,
        starts=starts,
        tasks=tasks,
        num_agents=args.num_agents,
        tasks_per_agent=args.tasks_per_agent,
        layout_str=layout,
    )

    print(
        f"[lacam_cpp] ticks={result['ticks']}, total_moves={result['total_moves']}, "
        f"total_completed={sum(result['completed_tasks']) if result['completed_tasks'] else 'n/a'}, "
        f"runtime={result['runtime']:.2f}s"
    )

if __name__ == "__main__":
    main()
