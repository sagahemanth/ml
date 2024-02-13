import numpy as np

def matrix_power(matrix, power):
    if not isinstance(matrix, np.ndarray) or len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input must be a square matrix")

    if not isinstance(power, int) or power < 0:
        raise ValueError("Power must be a non-negative integer")

    return np.linalg.matrix_power(matrix, power)

# Example usage
matrix_A = np.array([[2, 3], [4, 5]])
power_m = 3
result_matrix = matrix_power(matrix_A, power_m)

print(f"A^m: \n{result_matrix}")
