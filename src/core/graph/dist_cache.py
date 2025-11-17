from collections import deque

from core.graph.base import GraphBase

class DistCache:
    def __init__(self, graph: GraphBase):
        self.graph = graph
        self.cache = {}  # goal → dist[]

    def dist_map(self, goal: int):
        if goal in self.cache:
            return self.cache[goal]

        dist = [-1] * self.graph.num_vertices()
        dist[goal] = 0

        q = deque([goal])
        while q:
            v = q.popleft()
            for u in self.graph.neighbors(v):
                if dist[u] == -1:
                    dist[u] = dist[v] + 1
                    q.append(u)

        self.cache[goal] = dist
        return dist

    def dist(self, u, v):
        dm = self.dist_map(v)
        return dm[u]
