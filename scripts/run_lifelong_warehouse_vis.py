#!/usr/bin/env python3
"""
Helper to launch lifelong warehouse visualization with correctly ordered episodes.

It expects that you already ran:
    LACAM_VIS_EXPORT=1 pytest tests/test_lifelong_lacam.py::test_lifelong_warehouse_four_agents -q
which produces vis_outputs/test_lifelong_warehouse.map and ..._ep*.txt files.
"""

import glob
import os
import re
import shutil
import subprocess
import sys
from typing import List


def _episode_sort_key(path: str) -> int:
    m = re.search(r"_ep(\d+)\.txt$", os.path.basename(path))
    return int(m.group(1)) if m else 10**9


def main() -> int:
    vis_bin = shutil.which("mapf-visualizer-lifelong")
    if not vis_bin:
        print("mapf-visualizer-lifelong not found in PATH. Install it or adjust PATH.", file=sys.stderr)
        return 1

    vis_dir = os.environ.get("LACAM_VIS_DIR", "vis_outputs")
    map_path = os.path.join(vis_dir, "test_lifelong_warehouse.map")
    if not os.path.exists(map_path):
        print(f"Map file not found: {map_path}. Run the pytest exporter first.", file=sys.stderr)
        return 1

    episode_glob = os.path.join(vis_dir, "test_lifelong_warehouse_ep*.txt")
    episodes: List[str] = sorted(glob.glob(episode_glob), key=_episode_sort_key)
    episodes = [p for p in episodes if re.search(r"_ep\d+\.txt$", os.path.basename(p))]
    if not episodes:
        print(f"No episode files matched {episode_glob}. Run the pytest exporter first.", file=sys.stderr)
        return 1

    cmd = [vis_bin, map_path, *episodes]
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Visualizer failed: {exc}", file=sys.stderr)
        return exc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
