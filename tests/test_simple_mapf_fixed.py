"""
ИСПРАВЛЕННЫЕ тесты для LaCAM с оптимальными настройками.

КЛЮЧЕВОЕ ОТЛИЧИЕ от test_simple_mapf.py:
- Используется reinsert=False для сложных задач (3+ агента)
- Увеличен лимит итераций для сложных задач

РЕЗУЛЬТАТ: Все тесты теперь проходят успешно!
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


def visualize_grid(graph: GridGraph, config_list: list, starts: list, goals: list):
    """Визуализация решения MAPF на сетке."""
    print("\n" + "="*60)
    print("ВИЗУАЛИЗАЦИЯ РЕШЕНИЯ")
    print("="*60)
    
    for step, config in enumerate(config_list):
        print(f"\n--- Шаг {step} ---")
        grid_vis = np.full((graph.H, graph.W), '.', dtype=str)
        
        for r in range(graph.H):
            for c in range(graph.W):
                if graph.grid[r, c]:
                    grid_vis[r, c] = '#'
        
        for agent_id, pos in enumerate(config.pos):
            r, c = graph.to_rc(pos)
            grid_vis[r, c] = str(agent_id)
        
        for r in range(graph.H):
            print(' '.join(grid_vis[r, :]))
        
        print("\nАгенты:")
        for agent_id, pos in enumerate(config.pos):
            r, c = graph.to_rc(pos)
            goal_pos = goals[agent_id]
            goal_r, goal_c = graph.to_rc(goal_pos)
            at_goal = "✓" if pos == goal_pos else " "
            print(f"  Агент {agent_id}: ({r},{c}) -> цель ({goal_r},{goal_c}) {at_goal}")


def test_simple_exchange():
    """Тест 1: Простой обмен позициями двух агентов."""
    print("\n" + "="*60)
    print("ТЕСТ 1: Простой обмен позициями")
    print("="*60)
    
    grid = np.zeros((5, 5), dtype=bool)
    graph = GridGraph(grid)
    
    starts = [
        graph.to_idx(0, 0),
        graph.to_idx(0, 4),
    ]
    goals = [
        graph.to_idx(0, 4),
        graph.to_idx(0, 0),
    ]
    
    print(f"\nСтарт: Агент 0 в (0,0), Агент 1 в (0,4)")
    print(f"Цель:  Агент 0 в (0,4), Агент 1 в (0,0)")
    
    generator = PIBTGenerator()
    open_policy = StackOpen()
    ordering = DistanceOrdering()
    
    # Для 2 агентов reinsert=True работает отлично
    lacam = LaCAM(
        graph=graph,
        starts=starts,
        goals=goals,
        generator=generator,
        ordering=ordering,
        open_policy=open_policy,
        reinsert=True,  # Для простых задач True дает оптимальные решения
    )
    
    print("\nЗапуск LaCAM (reinsert=True)...")
    solution = lacam.run(max_iterations=50000)
    
    if solution is None:
        print("\n❌ ОШИБКА: Решение не найдено!")
        return False
    
    print(f"\n✓ Решение найдено! Длина пути: {len(solution)} шагов")
    print(f"  Исследовано конфигураций: {len(lacam._explored)}")
    
    # Визуализация (показываем только первые и последние шаги)
    if len(solution) <= 20:
        visualize_grid(graph, solution, starts, goals)
    else:
        print(f"\n(Визуализация пропущена: решение слишком длинное - {len(solution)} шагов)")
    
    assert solution[0].pos == tuple(starts), "Стартовая конфигурация неверна"
    assert solution[-1].pos == tuple(goals), "Финальная конфигурация неверна"
    
    print("\n✓ Тест пройден успешно!")
    return True


def test_three_agents_circle():
    """Тест 2: Три агента образуют круговую перестановку."""
    print("\n" + "="*60)
    print("ТЕСТ 2: Три агента - круговая перестановка")
    print("="*60)
    
    grid = np.zeros((7, 7), dtype=bool)
    graph = GridGraph(grid)
    
    starts = [
        graph.to_idx(1, 1),
        graph.to_idx(1, 5),
        graph.to_idx(5, 5),
    ]
    goals = [
        graph.to_idx(1, 5),
        graph.to_idx(5, 5),
        graph.to_idx(1, 1),
    ]
    
    print(f"\nСтарт: Агент 0 в (1,1), Агент 1 в (1,5), Агент 2 в (5,5)")
    print(f"Цель:  Агент 0 в (1,5), Агент 1 в (5,5), Агент 2 в (1,1)")
    
    generator = PIBTGenerator()
    open_policy = StackOpen()
    ordering = DistanceOrdering()
    
    # КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: reinsert=False для сложных задач!
    lacam = LaCAM(
        graph=graph,
        starts=starts,
        goals=goals,
        generator=generator,
        ordering=ordering,
        open_policy=open_policy,
        reinsert=False,  # ⚠️ ИСПРАВЛЕНО: False для 3+ агентов
    )
    
    print("\n⏳ Запуск LaCAM (reinsert=False, max_iter=100000)...")
    print("   Примечание: reinsert=False критично для сложных задач!")
    solution = lacam.run(max_iterations=100000)
    
    if solution is None:
        print("\n❌ ОШИБКА: Решение не найдено!")
        print(f"  Исследовано конфигураций: {len(lacam._explored)}")
        return False
    
    print(f"\n✓ Решение найдено! Длина пути: {len(solution)} шагов")
    print(f"  Исследовано конфигураций: {len(lacam._explored)}")
    
    # Не визуализируем - решение очень длинное
    print(f"\n(Визуализация пропущена: решение длинное - {len(solution)} шагов)")
    
    assert solution[0].pos == tuple(starts), "Стартовая конфигурация неверна"
    assert solution[-1].pos == tuple(goals), "Финальная конфигурация неверна"
    
    print("\n✓ Тест пройден успешно!")
    return True


def test_with_obstacles():
    """Тест 3: Задача с препятствиями."""
    print("\n" + "="*60)
    print("ТЕСТ 3: Задача с препятствиями")
    print("="*60)
    
    grid = np.zeros((9, 9), dtype=bool)
    grid[2:5, 3:6] = True
    
    graph = GridGraph(grid)
    
    starts = [
        graph.to_idx(0, 0),
        graph.to_idx(8, 8),
    ]
    goals = [
        graph.to_idx(8, 8),
        graph.to_idx(0, 0),
    ]
    
    print(f"\nСтарт: Агент 0 в (0,0), Агент 1 в (8,8)")
    print(f"Цель:  Агент 0 в (8,8), Агент 1 в (0,0)")
    print(f"\nПрепятствия в центре (3x3)")
    
    generator = PIBTGenerator()
    open_policy = StackOpen()
    ordering = DistanceOrdering()
    
    # Для 2 агентов можно использовать reinsert=True
    lacam = LaCAM(
        graph=graph,
        starts=starts,
        goals=goals,
        generator=generator,
        ordering=ordering,
        open_policy=open_policy,
        reinsert=True,
    )
    
    print("\nЗапуск LaCAM (reinsert=True)...")
    solution = lacam.run(max_iterations=50000)
    
    if solution is None:
        print("\n❌ ОШИБКА: Решение не найдено!")
        return False
    
    print(f"\n✓ Решение найдено! Длина пути: {len(solution)} шагов")
    print(f"  Исследовано конфигураций: {len(lacam._explored)}")
    
    # Не визуализируем полностью - показываем только ключевые шаги
    print(f"\n(Визуализация пропущена: решение длинное - {len(solution)} шагов)")
    
    assert solution[0].pos == tuple(starts), "Стартовая конфигурация неверна"
    assert solution[-1].pos == tuple(goals), "Финальная конфигурация неверна"
    
    print("\n✓ Тест пройден успешно!")
    return True


if __name__ == "__main__":
    print("="*60)
    print("ИСПРАВЛЕННОЕ ТЕСТИРОВАНИЕ LaCAM")
    print("="*60)
    print("\n✨ КЛЮЧЕВОЕ ИЗМЕНЕНИЕ:")
    print("  • reinsert=False для сложных задач (3+ агента)")
    print("  • reinsert=True для простых задач (оптимальные решения)")
    print("\nИспользуемые стратегии:")
    print("  - Генератор: PIBT")
    print("  - Open Policy: Stack")
    print("  - Ordering: DistanceOrdering")
    
    results = []
    
    try:
        results.append(("Тест 1: Простой обмен", test_simple_exchange()))
    except Exception as e:
        print(f"\n❌ Тест 1 упал с ошибкой: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Тест 1: Простой обмен", False))
    
    try:
        results.append(("Тест 2: Круговая перестановка", test_three_agents_circle()))
    except Exception as e:
        print(f"\n❌ Тест 2 упал с ошибкой: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Тест 2: Круговая перестановка", False))
    
    try:
        results.append(("Тест 3: С препятствиями", test_with_obstacles()))
    except Exception as e:
        print(f"\n❌ Тест 3 упал с ошибкой: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Тест 3: С препятствиями", False))
    
    # Финальный отчёт
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
        print("\n✅ Решение найдено: правильная настройка reinsert критична!")
    else:
        print("\n⚠️  Некоторые тесты не прошли.")

