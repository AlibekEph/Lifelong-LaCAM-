"""
Тест для Lifelong LaCAM со встроенной логикой обновления целей.

КЛЮЧЕВОЕ ОТЛИЧИЕ от test_lifelong.py:
- НЕТ внешнего цикла replanning
- Логика Lifelong встроена В САМ АЛГОРИТМ LaCAM
- Один вызов run() выполняет весь Lifelong MAPF
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from core.graph.grid import GridGraph
from core.lifelong_lacam_integrated import LifelongLaCAMIntegrated
from strategies.generators.pibt import PIBTGenerator
from strategies.open_policy.stack import StackOpen
from strategies.ordering.distance_ordering import DistanceOrdering


def test_integrated_simple():
    """
    Простой тест встроенной версии:
    - 2 агента циклически меняются углами
    - Один вызов run() выполняет всё
    """
    print("\n" + "="*70)
    print("ТЕСТ: Встроенная логика Lifelong (2 агента)")
    print("="*70)
    
    grid = np.zeros((5, 5), dtype=bool)
    graph = GridGraph(grid)
    
    starts = [
        graph.to_idx(0, 0),
        graph.to_idx(4, 4),
    ]
    
    initial_goals = [
        graph.to_idx(4, 0),
        graph.to_idx(0, 4),
    ]
    
    # Циклические цели
    corners = [
        graph.to_idx(0, 0),
        graph.to_idx(0, 4),
        graph.to_idx(4, 0),
        graph.to_idx(4, 4),
    ]
    task_indices = [0, 0]
    
    def assign_task(agent_id: int, current_pos: int, old_goal: int) -> int:
        """Циклически назначаем углы."""
        task_indices[agent_id] = (task_indices[agent_id] + 1) % len(corners)
        new_goal = corners[task_indices[agent_id]]
        
        # Убедимся что не совпадает с текущей позицией
        while new_goal == current_pos:
            task_indices[agent_id] = (task_indices[agent_id] + 1) % len(corners)
            new_goal = corners[task_indices[agent_id]]
        
        return new_goal
    
    print(f"\nСтарт: (0,0) и (4,4)")
    print(f"Начальные цели: (4,0) и (0,4)")
    print(f"Callback: циклически по углам")
    print(f"Условие остановки: 3 задачи на агента")
    
    lifelong = LifelongLaCAMIntegrated(
        graph=graph,
        starts=starts,
        initial_goals=initial_goals,
        generator=PIBTGenerator(),
        ordering=DistanceOrdering(),
        open_policy=StackOpen(),
        task_callback=assign_task,
        reinsert=False,
        max_tasks_per_agent=3,  # остановимся после 3 задач на агента
    )
    
    print("\n⏳ Запуск ОДНОГО run() со встроенной логикой...")
    print("   (обновления целей происходят ВНУТРИ алгоритма)")
    
    solution = lifelong.run(max_iterations=100000, verbose=True)
    
    if solution:
        print(f"\n✅ РЕШЕНИЕ НАЙДЕНО!")
        print(f"   Длина пути: {len(solution)} шагов")
    else:
        print(f"\n⚠️  Решение не найдено")
    
    stats = lifelong.get_statistics()
    
    print(f"\n📊 Статистика:")
    print(f"   Всего итераций LaCAM: {stats['total_iterations']}")
    print(f"   Обновлений целей: {stats['goal_updates']}")
    print(f"   Задач выполнено: {stats['total_completed_tasks']}")
    print(f"   Задач на агента: {stats['completed_tasks_per_agent']}")
    
    if solution:
        # Проверка корректности
        assert solution[0].pos == tuple(starts), "Стартовая конфигурация неверна"
        assert stats['total_completed_tasks'] >= 6, "Должно быть минимум 6 задач (3 на агента)"
        
        print(f"\n✓ Встроенная версия работает!")
        return True
    
    return False


def test_integrated_warehouse():
    """
    Сценарий склада со встроенной логикой.
    """
    print("\n" + "="*70)
    print("ТЕСТ: Склад со встроенной логикой (3 робота)")
    print("="*70)
    
    grid = np.zeros((7, 7), dtype=bool)
    graph = GridGraph(grid)
    
    starts = [
        graph.to_idx(0, 0),
        graph.to_idx(0, 1),
        graph.to_idx(0, 2),
    ]
    
    delivery_points = [
        graph.to_idx(6, 6),
        graph.to_idx(6, 3),
        graph.to_idx(6, 0),
        graph.to_idx(3, 6),
        graph.to_idx(3, 0),
    ]
    
    initial_goals = [
        delivery_points[0],
        delivery_points[1],
        delivery_points[2],
    ]
    
    import random
    random.seed(42)
    
    def warehouse_task(agent_id: int, current_pos: int, old_goal: int) -> int:
        """Назначаем случайную точку доставки."""
        available = [p for p in delivery_points if p != current_pos]
        return random.choice(available) if available else delivery_points[0]
    
    print(f"\n🤖 3 робота на складе")
    print(f"📦 Динамические доставки через встроенный callback")
    print(f"🛑 Остановка: 2 доставки на робота")
    
    lifelong = LifelongLaCAMIntegrated(
        graph=graph,
        starts=starts,
        initial_goals=initial_goals,
        generator=PIBTGenerator(),
        ordering=DistanceOrdering(),
        open_policy=StackOpen(),
        task_callback=warehouse_task,
        reinsert=False,
        max_tasks_per_agent=2,
    )
    
    print("\n⏳ Один непрерывный run() LaCAM...")
    
    solution = lifelong.run(max_iterations=150000, verbose=False)
    
    stats = lifelong.get_statistics()
    
    print(f"\n📊 Результаты:")
    print(f"   Итераций: {stats['total_iterations']}")
    print(f"   Обновлений целей: {stats['goal_updates']}")
    print(f"   Всего доставок: {stats['total_completed_tasks']}")
    print(f"   На робота: {stats['completed_tasks_per_agent']}")
    
    if solution:
        print(f"   Длина пути: {len(solution)} шагов")
        print(f"\n✓ Склад работает!")
        return True
    else:
        print(f"\n⚠️  Не удалось найти решение за {stats['total_iterations']} итераций")
        return False


def test_comparison_replanning_vs_integrated():
    """
    Сравнение двух подходов:
    1. Внешний replanning (test_lifelong.py)
    2. Встроенная логика (этот файл)
    """
    print("\n" + "="*70)
    print("СРАВНЕНИЕ: Replanning vs Встроенная логика")
    print("="*70)
    
    grid = np.zeros((5, 5), dtype=bool)
    graph = GridGraph(grid)
    
    starts = [
        graph.to_idx(0, 0),
        graph.to_idx(4, 4),
    ]
    
    initial_goals = [
        graph.to_idx(4, 0),
        graph.to_idx(0, 4),
    ]
    
    corners = [
        graph.to_idx(0, 0),
        graph.to_idx(0, 4),
        graph.to_idx(4, 0),
        graph.to_idx(4, 4),
    ]
    
    print("\n--- Встроенная логика ---")
    
    task_indices = [0, 0]
    
    def assign_task(agent_id: int, current_pos: int, old_goal: int) -> int:
        task_indices[agent_id] = (task_indices[agent_id] + 1) % len(corners)
        new_goal = corners[task_indices[agent_id]]
        while new_goal == current_pos:
            task_indices[agent_id] = (task_indices[agent_id] + 1) % len(corners)
            new_goal = corners[task_indices[agent_id]]
        return new_goal
    
    lifelong = LifelongLaCAMIntegrated(
        graph=graph,
        starts=starts,
        initial_goals=initial_goals,
        generator=PIBTGenerator(),
        ordering=DistanceOrdering(),
        open_policy=StackOpen(),
        task_callback=assign_task,
        reinsert=False,
        max_tasks_per_agent=2,
    )
    
    solution = lifelong.run(max_iterations=50000, verbose=False)
    stats = lifelong.get_statistics()
    
    print(f"  Итераций LaCAM: {stats['total_iterations']}")
    print(f"  Обновлений целей: {stats['goal_updates']}")
    print(f"  Задач выполнено: {stats['total_completed_tasks']}")
    print(f"  Результат: {'✓ Решение найдено' if solution else '✗ Нет решения'}")
    
    print("\n💡 Ключевые отличия:")
    print("  Replanning подход:")
    print("    • Внешний цикл")
    print("    • Множество вызовов LaCAM")
    print("    • Может застревать между replanning'ами")
    print("\n  Встроенная логика:")
    print("    • Один непрерывный run()")
    print("    • Обновления целей ВНУТРИ алгоритма")
    print("    • Плавное продолжение поиска с новыми целями")
    
    return solution is not None


if __name__ == "__main__":
    print("="*70)
    print(" "*10 + "LIFELONG LaCAM СО ВСТРОЕННОЙ ЛОГИКОЙ")
    print("="*70)
    print("\n🎯 Логика обновления целей встроена В САМ АЛГОРИТМ LaCAM")
    print("   Один вызов run() выполняет весь Lifelong MAPF")
    
    results = []
    
    try:
        results.append(("Встроенная простая", test_integrated_simple()))
    except Exception as e:
        print(f"\n❌ Тест упал: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Встроенная простая", False))
    
    try:
        results.append(("Склад встроенный", test_integrated_warehouse()))
    except Exception as e:
        print(f"\n❌ Тест упал: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Склад встроенный", False))
    
    try:
        results.append(("Сравнение подходов", test_comparison_replanning_vs_integrated()))
    except Exception as e:
        print(f"\n❌ Тест упал: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Сравнение подходов", False))
    
    print("\n" + "="*70)
    print("ИТОГОВЫЙ ОТЧЁТ")
    print("="*70)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    passed_count = sum(1 for _, p in results if p)
    print(f"\nПройдено: {passed_count}/{len(results)}")
    
    if passed_count == len(results):
        print("\n🎉 Встроенная версия Lifelong LaCAM работает!")
        print("\n✅ Преимущества:")
        print("   • Логика в самом алгоритме LaCAM")
        print("   • Один непрерывный run()")
        print("   • Обновление целей по ходу генерации конфигураций")
        print("   • Нет внешнего цикла replanning")

