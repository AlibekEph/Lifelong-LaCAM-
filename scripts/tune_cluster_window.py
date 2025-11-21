"""
Перебор окон кластеризации (w) для Lifelong LaCAM.

Пример:
    python3 scripts/tune_cluster_window.py --windows 1 2 3
"""

import os
import sys
import json
import argparse
from pathlib import Path
import random
import numpy as np

# добавляем корень репозитория в sys.path и src для импортов
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tests.test_lifelong_warehouse_512_bulk import (  # type: ignore
    run_lacam,
    sample_unique_free,
    WAREHOUSE_MAP_PATH,
)
from utils.kiva_loader import layout_to_grid, generate_kiva_tasks
from core.graph.grid import GridGraph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Подбор окна кластеризации Lifelong LaCAM")
    parser.add_argument("--map-json", type=Path, default=WAREHOUSE_MAP_PATH, help="Путь до json с раскладкой склада")
    parser.add_argument("--windows", type=int, nargs="+", default=[1, 2, 3], help="Список w для перебора")
    parser.add_argument("--num-agents", type=int, default=64, help="Количество агентов")
    parser.add_argument("--tasks-per-agent", type=int, default=200, help="Число задач на агента")
    parser.add_argument("--seed", type=int, default=7, help="Сид генерации стартов/задач")
    parser.add_argument("--stop-mode", choices=["tasks", "ticks"], default="tasks", help="Правило остановки")
    parser.add_argument("--ticks-limit", type=int, default=20000, help="Лимит тиков при stop-mode=ticks")
    parser.add_argument("--no-clustering", action="store_true", help="Отключить кластеризацию")
    parser.add_argument("--priority-open", action="store_true", help="Использовать CompletedPriorityOpen")
    parser.add_argument("--min-goal-distance", type=int, default=6, help="Минимальная дистанция между целями подряд")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert args.map_json.exists(), f"Не найден файл карты: {args.map_json}"

    payload = json.loads(args.map_json.read_text())
    grid = layout_to_grid(payload["layout"])
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
        min_goal_distance=args.min_goal_distance,
    )

    stop_value = args.tasks_per_agent if args.stop_mode == "tasks" else args.ticks_limit

    print("=== Cluster window sweep ===")
    for w in args.windows:
        result = run_lacam(
            graph=graph,
            starts=starts,
            tasks=tasks,
            stop_mode=args.stop_mode,
            stop_value=stop_value,
            enable_clustering=not args.no_clustering,
            use_priority_open=args.priority_open,
            cluster_window=w,
        )
        total_completed = sum(result["completed_tasks"]) if result["completed_tasks"] else 0
        print(
            f"w={w}: ticks={result['ticks']}, total_moves={result['total_moves']}, "
            f"completed={total_completed}, runtime={result['runtime']:.2f}s, note={result.get('note')}"
        )


if __name__ == "__main__":
    main()
