import unittest
from jacobi_method import jacobi_eigen

class TestJacobiMethod(unittest.TestCase):
    
    def assertMatrixClose(self, mat1, mat2, tol=1e-5):
        """Вспомогательный метод для сравнения матриц с учетом погрешности."""
        self.assertEqual(len(mat1), len(mat2))
        for i in range(len(mat1)):
            self.assertEqual(len(mat1[i]), len(mat2[i]))
            for j in range(len(mat1[i])):
                self.assertAlmostEqual(mat1[i][j], mat2[i][j], delta=tol)

    def test_2x2_matrix(self):
        """Тестирование корректной симметричной матрицы 2x2."""
        A = [
            [2.0, 1.0],
            [1.0, 2.0]
        ]
        eigenvalues, eigenvectors = jacobi_eigen(A)
        # Ожидаемые собственные числа: 3 и 1
        eigenvalues.sort(reverse=True)
        self.assertAlmostEqual(eigenvalues[0], 3.0, delta=1e-5)
        self.assertAlmostEqual(eigenvalues[1], 1.0, delta=1e-5)

    def test_diagonal_matrix(self):
        """Тестирование диагональной матрицы (уже содержит собственные значения)."""
        A = [
            [5.0, 0.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.0, 0.0, 4.0]
        ]
        eigenvalues, _ = jacobi_eigen(A)
        expected = sorted([5.0, -2.0, 4.0])
        self.assertEqual(sorted(eigenvalues), expected)

    def test_invalid_empty_matrix(self):
        """Тестирование реакции на пустую матрицу."""
        with self.assertRaisesRegex(ValueError, "не может быть пустой"):
            jacobi_eigen([])

    def test_invalid_non_square_matrix(self):
        """Тестирование реакции на неквадратную матрицу."""
        A = [
            [1.0, 2.0],
            [2.0, 1.0, 3.0]
        ]
        with self.assertRaisesRegex(ValueError, "должна быть квадратной"):
            jacobi_eigen(A)

    def test_invalid_asymmetric_matrix(self):
        """Тестирование реакции на асимметричную матрицу."""
        A = [
            [1.0, 2.0],
            [3.0, 4.0]
        ]
        with self.assertRaisesRegex(ValueError, "должна быть симметричной"):
            jacobi_eigen(A)
            
    def test_max_iterations_error(self):
        """Тестирование предотвращения зацикливания."""
        A = [
            [2.0, 1.0],
            [1.0, 2.0]
        ]
        # Принудительно ставим 0 итераций, чтобы вызвать ошибку нехватки шагов
        with self.assertRaisesRegex(RuntimeError, "Алгоритм не сошелся"):
            jacobi_eigen(A, max_iterations=0)

if __name__ == '__main__':
    unittest.main()