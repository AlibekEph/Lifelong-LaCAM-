from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class Constraint:
    """
    LL-node в LaCAM (low-level constraint node).

    Хранит:
        parent : Constraint | None
            Родительский узел LL-дерева.

        who : int | None
            Какой агент должен сделать этот шаг.
            None используется только в корне.

        where : int | None
            В какую вершину агент должен перейти.
            None в корне.

        depth : int
            Глубина узла от корня LL-дерева.

    Эта структура *неизменяемая логически*, но не frozen, чтобы
    можно было быстро инициализировать LL-узлы большого числа.
    """

    parent: Optional["Constraint"]
    who: Optional[int]
    where: Optional[int]
    depth: int

    def is_root(self) -> bool:
        """True если это корневой LL-узел (нет родителя)."""
        return self.parent is None

    def path(self):
        """
        Восстановить путь ограничений от корня до этого узла.
        Редко используется, но полезно для отладки.
        """
        node = self
        chain = []
        while node is not None:
            chain.append(node)
            node = node.parent
        return list(reversed(chain))

    def __repr__(self):
        if self.is_root():
            return "Constraint(root)"
        return f"Constraint(who={self.who}, where={self.where}, depth={self.depth})"
