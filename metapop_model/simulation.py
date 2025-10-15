
import numpy as np
from scipy.sparse import linalg as splinalg

# La función `create_delta_t_matrix` ha sido eliminada porque causaba el error de memoria.

def run_simulation(H, initial_vector, delta_t, num_steps):
    """
    Ejecuta una simulación iterativa aplicando la acción de la matriz exponencial en cada paso.
    Calcula P(t_i+1) = expm(H * delta_t) * P(t_i) para cada paso i.
    Este método es eficiente en memoria porque no construye la matriz expm(H * delta_t) completa.

    Args:
        H (scipy.sparse.spmatrix): El Hamiltoniano (matriz de transición continua).
        initial_vector (np.ndarray): El vector de estado inicial P(0).
        delta_t (float): El paso de tiempo para cada iteración.
        num_steps (int): El número de pasos de tiempo a simular.

    Returns:
        list[np.ndarray]: Una lista con el vector de estado en cada paso de tiempo.
    """
    history = [initial_vector]
    current_vector = initial_vector
    for i in range(num_steps):
        # expm_multiply calcula la acción de la matriz exponencial sobre un vector
        # sin construir la matriz completa. Es mucho más eficiente en memoria.
        current_vector = splinalg.expm_multiply(H * delta_t, current_vector)
        history.append(current_vector)
        # Opcional: imprimir progreso para simulaciones largas
        if (i + 1) % 20 == 0:
            print(f"  ... Simulación: paso {i+1}/{num_steps} completado")
            
    return history
