import numpy as np
from numpy.typing import NDArray
from typing import List
from collections import deque

from core.graph.base import GraphBase

class GridGraph(GraphBase):
    def __init__(self, grid: NDArray[np.bool_]):
        """
        Инициализируем сетку в виде матрицы bool, где:
        grid[y, x] == True → препятствие (стена)
        grid[y, x] == False → свободная клетка
        """
        self.grid = grid.astype(np.bool_)
        self.H, self.W = grid.shape

        # число вершин
        self.V = self.H * self.W

        # заранее вычисляем соседей
        self._neighbors = self._compute_neighbors()
        # Кэш расстояний: goal -> dist_map
        self._dist_cache: dict[int, list[int]] = {}
    
    def _compute_neighbors(self):
        neigh = [[] for _ in range(self.V)]
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]

        for r in range(self.H):
            for c in range(self.W):
                if self.grid[r, c]:
                    continue  # стена
                v = self.to_idx(r, c)

                for dr,dc in dirs:
                    rr,cc = r+dr, c+dc
                    if 0 <= rr < self.H and 0 <= cc < self.W and not self.grid[rr, cc]:
                        neigh[v].append(self.to_idx(rr,cc))
        return neigh

    def neighbors(self, v: int) -> List[int]:
        return self._neighbors[v]

    def num_vertices(self) -> int:
        return self.V

    def is_blocked(self, v):
        r, c = self.to_rc(v)
        return self.grid[r, c]

    # Утилиты для преобразования между координатами и индексами вершин
    def to_idx(self, r: int, c: int) -> int:
        return r * self.W + c

    # Утилита для преобразования индекса вершины в координаты
    def to_rc(self, v: int) -> tuple[int, int]:
        return v // self.W, v % self.W

    def dist(self, u: int, v: int) -> int:
        """
        Кратчайшее расстояние (BFS) между вершинами u и v.
        Возвращает -1, если цель недостижима.
        """
        if u == v:
            return 0

        dist_map = self._dist_cache.get(v)
        if dist_map is None:
            dist_map = [-1] * self.V
            dist_map[v] = 0
            q = deque([v])
            while q:
                curr = q.popleft()
                for neighbor in self._neighbors[curr]:
                    if dist_map[neighbor] == -1:
                        dist_map[neighbor] = dist_map[curr] + 1
                        q.append(neighbor)
            self._dist_cache[v] = dist_map

        return dist_map[u]
        
