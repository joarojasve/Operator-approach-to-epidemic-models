
import numpy as np
from scipy.sparse import linalg as splinalg

def create_delta_t_matrix(H, delta_t):
    """
    Calcula la matriz de transición de paso de tiempo P(delta_t) = expm(H * delta_t).

    Args:
        H (scipy.sparse.spmatrix): El Hamiltoniano (matriz de transición continua).
        delta_t (float): El paso de tiempo.

    Returns:
        scipy.sparse.spmatrix: La matriz de transición discreta P(delta_t).
    """
    # expm calcula la exponencial de la matriz, que puede ser densa.
    # Se devuelve como una matriz densa de numpy o una matriz dispersa si es apropiado.
    return splinalg.expm(H * delta_t)

def run_simulation(P_delta_t, initial_vector, num_steps):
    """
    Ejecuta una simulación iterativa aplicando la matriz P(delta_t).

    Args:
        P_delta_t (scipy.sparse.spmatrix): La matriz de transición de paso de tiempo.
        initial_vector (np.ndarray): El vector de estado inicial P(0).
        num_steps (int): El número de pasos de tiempo a simular.

    Returns:
        list[np.ndarray]: Una lista con el vector de estado en cada paso de tiempo.
    """
    history = [initial_vector]
    current_vector = initial_vector
    for _ in range(num_steps):
        current_vector = P_delta_t.dot(current_vector)
        history.append(current_vector)
    return history
