import math
from typing import Tuple, List

def jacobi_eigen(matrix: List[List[float]], epsilon: float = 1e-9, max_iterations: int = 1000) -> Tuple[List[float], List[List[float]]]:
    """
    Вычисляет собственные значения и собственные векторы действительной симметричной матрицы методом Якоби.

    Аргументы:
        matrix: Действительная симметричная матрица (список списков).
        epsilon: Требуемая точность (максимально допустимая сумма квадратов недиагональных элементов).
        max_iterations: Максимальное количество итераций (для предотвращения зацикливания).

    Возвращает:
        Кортеж из двух элементов:
        1. Список собственных чисел.
        2. Матрица собственных векторов (где каждый столбец - это вектор).

    Исключения:
        ValueError: Если матрица пуста, не квадратная или не симметричная.
        RuntimeError: Если алгоритм не сошелся за отведенное число итераций.
    """
    # --- 1. Механизм контрактов (Валидация исходных данных) ---
    n = len(matrix)
    if n == 0:
        raise ValueError("Матрица не может быть пустой.")
    for row in matrix:
        if len(row) != n:
            raise ValueError("Матрица должна быть квадратной.")
    
    # Проверка на симметричность
    for i in range(n):
        for j in range(n):
            if abs(matrix[i][j] - matrix[j][i]) > 1e-10:
                raise ValueError("Матрица должна быть симметричной.")

    # --- 2. Инициализация локальных переменных ---
    # A_i: Текущая матрица (копируем исходную)
    A = [[float(val) for val in row] for row in matrix]
    
    # V_i: Текущая матрица собственных векторов (начинается с единичной)
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    # S_i: Сумма квадратов недиагональных элементов.
    # Для оптимизации считаем только верхний треугольник (так как матрица симметрична)
    S = sum(A[i][j]**2 for i in range(n) for j in range(i + 1, n))
    
    iterations = 0

    # --- 3. Шаги метода Якоби ---
    while S > epsilon and iterations < max_iterations:
        iterations += 1

        # (a) Выбираем зануляемый недиагональный элемент Apq.
        # Выбираем максимальный по модулю элемент для максимально быстрой сходимости
        p, q = 0, 1
        max_val = -1.0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A[i][j]) > max_val:
                    max_val = abs(A[i][j])
                    p = i
                    q = j

        if max_val**2 < epsilon: 
            break # Практическая остановка, если максимальный элемент уже слишком мал

        Apq = A[p][q]
        App = A[p][p]
        Aqq = A[q][q]

        # (b) Вычисляем величину C
        C = (Aqq - App) / (2.0 * Apq)

        # (c) Вычисляем тангенс угла поворота
        if C >= 0:
            t = 1.0 / (C + math.sqrt(C**2 + 1.0))
        else:
            t = 1.0 / (C - math.sqrt(C**2 + 1.0))

        # (d) Вычисляем синус и косинус угла поворота
        c = 1.0 / math.sqrt(1.0 + t**2)
        s = t * c

        # (e) Обновляем элементы матрицы A
        A[p][q] = A[q][p] = 0.0
        A[p][p] = App - Apq * t
        A[q][q] = Aqq + Apq * t

        for r in range(n):
            if r != p and r != q:
                Arp = A[r][p]
                Arq = A[r][q]

                term = s / (1.0 + c)
                A_rp_new = Arp - s * (Arq + term * Arp)
                A_rq_new = Arq + s * (Arp - term * Arq)

                A[r][p] = A[p][r] = A_rp_new
                A[r][q] = A[q][r] = A_rq_new

        # (f) Обновляем элементы матрицы собственных векторов V
        for r in range(n):
            Vrp = V[r][p]
            Vrq = V[r][q]
            V[r][p] = Vrp * c - Vrq * s
            V[r][q] = Vrp * s + Vrq * c

        # (g) Вычисляем новое значение суммы квадратов
        S = S - Apq**2
        if S < 0: # Корректировка возможной погрешности с плавающей точкой
            S = 0.0

    # --- 4. Условие остановки по зацикливанию ---
    if iterations >= max_iterations:
        raise RuntimeError(f"Алгоритм не сошелся за {max_iterations} итераций (вероятное зацикливание).")

    # --- 5. Выход ---
    eigenvalues = [A[i][i] for i in range(n)]
    return eigenvalues, V