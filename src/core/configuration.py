from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Iterable, List


@dataclass(frozen=True)
class Configuration:
    """
    Конфигурация MAPF: позиции всех агентов в один момент времени.

    Хранится как tuple[int], где pos[i] = вершина агента i.
    Immutable → можно безопасно класть в dict / set.
    """
    pos: Tuple[int, ...]   # позиции агентов

    # ------------------------------------------------------------
    # Доступ к данным
    # ------------------------------------------------------------
    def __getitem__(self, agent_id: int) -> int:
        """Позиция конкретного агента."""
        return self.pos[agent_id]

    def num_agents(self) -> int:
        """Количество агентов."""
        return len(self.pos)

    def apply_moves(self, moves: Iterable[int]) -> Configuration:
        """
        Создать новую конфигурацию, применив шаг каждого агента.
        moves[i] — новая вершина для агента i.

        Пример:
            new_conf = old_conf.apply_moves([5, 7, 2, 9])
        """
        # moves может быть list или tuple
        return Configuration(tuple(moves))

    def as_list(self) -> List[int]:
        """Получить list[int] позиции агентов."""
        return list(self.pos)

    def __len__(self) -> int:
        """Число агентов."""
        return len(self.pos)

    def __repr__(self) -> str:
        return f"Configuration(pos={self.pos})"
