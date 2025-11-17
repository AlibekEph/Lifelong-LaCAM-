
from core.graph.base import GraphBase
from typing import Dict, List

class GeneralGraph(GraphBase):
    def __init__(self, adjacency_list: dict[int, list[int]]):
        self.adj = adjacency_list

    def neighbors(self, v: int):
        return self.adj.get(v, [])

    def num_vertices(self):
        return len(self.adj)

    def is_blocked(self, v):
        return False