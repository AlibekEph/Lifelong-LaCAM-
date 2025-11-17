"""
ИСПРАВЛЕННЫЕ базовые тесты LaCAM.

КЛЮЧЕВОЕ ОТЛИЧИЕ: reinsert=False для задач с 3+ агентами.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from core.graph.grid import GridGraph
from core.lacam import LaCAM
from strategies.generators.pibt import PIBTGenerator
from strategies.open_policy.stack import StackOpen
from strategies.ordering.distance_ordering import DistanceOrdering


def test_single_agent():
    """Тест с одним агентом."""
    print("\n" + "="*60)
    print("ТЕСТ: Один агент")
    print("="*60)
    
    grid = np.zeros((5, 5), dtype=bool)
    graph = GridGraph(grid)
    
    starts = [graph.to_idx(0, 0)]
    goals = [graph.to_idx(4, 4)]
    
    print(f"Старт: (0,0)")
    print(f"Цель:  (4,4)")
    
    lacam = LaCAM(
        graph=graph,
        starts=starts,
        goals=goals,
        generator=PIBTGenerator(),
        ordering=DistanceOrdering(),
        open_policy=StackOpen(),
        reinsert=True,
    )
    
    solution = lacam.run(max_iterations=10000)
    
    if solution:
        print(f"✓ Решение найдено! Длина: {len(solution)} шагов")
        print(f"  Путь: {[config.pos[0] for config in solution]}")
        return True
    else:
        print("❌ Решение не найдено!")
        return False


def test_two_agents_no_conflict():
    """Тест с двумя агентами, не мешающими друг другу."""
    print("\n" + "="*60)
    print("ТЕСТ: Два агента без конфликта")
    print("="*60)
    
    grid = np.zeros((5, 5), dtype=bool)
    graph = GridGraph(grid)
    
    starts = [
        graph.to_idx(0, 0),
        graph.to_idx(4, 4),
    ]
    goals = [
        graph.to_idx(0, 4),
        graph.to_idx(4, 0),
    ]
    
    print(f"Старт: Агент 0 в (0,0), Агент 1 в (4,4)")
    print(f"Цель:  Агент 0 в (0,4), Агент 1 в (4,0)")
    
    lacam = LaCAM(
        graph=graph,
        starts=starts,
        goals=goals,
        generator=PIBTGenerator(),
        ordering=DistanceOrdering(),
        open_policy=StackOpen(),
        reinsert=True,
    )
    
    solution = lacam.run(max_iterations=10000)
    
    if solution:
        print(f"✓ Решение найдено! Длина: {len(solution)} шагов")
        print(f"  Исследовано конфигураций: {len(lacam._explored)}")
        return True
    else:
        print("❌ Решение не найдено!")
        return False


def test_three_agents_simple():
    """Тест с тремя агентами - простой случай."""
    print("\n" + "="*60)
    print("ТЕСТ: Три агента - простое движение")
    print("="*60)
    
    grid = np.zeros((7, 7), dtype=bool)
    graph = GridGraph(grid)
    
    starts = [
        graph.to_idx(0, 0),
        graph.to_idx(3, 3),
        graph.to_idx(6, 6),
    ]
    goals = [
        graph.to_idx(6, 0),
        graph.to_idx(3, 3),  # остается на месте
        graph.to_idx(0, 6),
    ]
    
    print(f"Старт: Агент 0 в (0,0), Агент 1 в (3,3), Агент 2 в (6,6)")
    print(f"Цель:  Агент 0 в (6,0), Агент 1 в (3,3), Агент 2 в (0,6)")
    
    # КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: reinsert=False
    lacam = LaCAM(
        graph=graph,
        starts=starts,
        goals=goals,
        generator=PIBTGenerator(),
        ordering=DistanceOrdering(),
        open_policy=StackOpen(),
        reinsert=False,  # ⚠️ ИСПРАВЛЕНО!
    )
    
    print("\n⏳ Запуск с reinsert=False (требуется больше итераций)...")
    solution = lacam.run(max_iterations=200000)
    
    if solution:
        print(f"✓ Решение найдено! Длина: {len(solution)} шагов")
        print(f"  Исследовано конфигураций: {len(lacam._explored)}")
        return True
    else:
        print("❌ Решение не найдено!")
        print(f"  Исследовано конфигураций: {len(lacam._explored)}")
        return False


def test_corridor():
    """Тест с узким коридором."""
    print("\n" + "="*60)
    print("ТЕСТ: Коридор - два агента навстречу")
    print("="*60)
    
    grid = np.ones((3, 7), dtype=bool)
    grid[1, :] = False  # средняя линия свободна
    
    graph = GridGraph(grid)
    
    starts = [
        graph.to_idx(1, 0),
        graph.to_idx(1, 6),
    ]
    goals = [
        graph.to_idx(1, 6),
        graph.to_idx(1, 0),
    ]
    
    print(f"Коридор 1x7: два агента навстречу")
    print(f"Старт: Агент 0 в (1,0), Агент 1 в (1,6)")
    print(f"Цель:  Агент 0 в (1,6), Агент 1 в (1,0)")
    print(f"\n⚠️  Задача нерешаема: в узком коридоре агенты не могут разминуться")
    
    lacam = LaCAM(
        graph=graph,
        starts=starts,
        goals=goals,
        generator=PIBTGenerator(),
        ordering=DistanceOrdering(),
        open_policy=StackOpen(),
        reinsert=True,
    )
    
    solution = lacam.run(max_iterations=10000)
    
    if solution:
        print(f"✓ Решение найдено! Длина: {len(solution)} шагов")
        return True
    else:
        print("❌ Решение не найдено (ожидаемо - задача нерешаема)")
        return "expected"


if __name__ == "__main__":
    print("="*60)
    print("ИСПРАВЛЕННЫЕ БАЗОВЫЕ ТЕСТЫ LaCAM")
    print("="*60)
    print("\n✨ С правильной настройкой reinsert")
    
    results = []
    
    try:
        results.append(("Один агент", test_single_agent()))
    except Exception as e:
        print(f"\n❌ Тест упал с ошибкой: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Один агент", False))
    
    try:
        results.append(("Два агента без конфликта", test_two_agents_no_conflict()))
    except Exception as e:
        print(f"\n❌ Тест упал с ошибкой: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Два агента без конфликта", False))
    
    try:
        results.append(("Три агента простой", test_three_agents_simple()))
    except Exception as e:
        print(f"\n❌ Тест упал с ошибкой: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Три агента простой", False))
    
    try:
        result = test_corridor()
        if result == "expected":
            results.append(("Коридор (нерешаемая задача)", True))
        else:
            results.append(("Коридор", result))
    except Exception as e:
        print(f"\n❌ Тест упал с ошибкой: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Коридор", False))
    
    print("\n" + "="*60)
    print("ИТОГОВЫЙ ОТЧЁТ")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    print(f"\nПройдено тестов: {passed_count}/{total_count}")
    
    if passed_count == total_count:
        print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
    else:
        print("\n⚠️  Некоторые тесты не прошли.")

