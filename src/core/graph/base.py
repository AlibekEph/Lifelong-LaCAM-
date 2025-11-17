from typing import Protocol, List

class GraphBase(Protocol):
    def neighbors(self, v: int) -> List[int]:
        """Возвращает список смежных вершин из v."""
        ...

    def num_vertices(self) -> int:
        """Количество вершин в графе."""
        ...

    def is_blocked(self, v: int) -> bool:
        """True — если вершина непроходима (стена), False — если проходима."""
        ...

    def dist(self, u: int, v: int) -> int:
        """Кратчайшее расстояние (BFS)."""
        ...
