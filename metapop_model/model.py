
import numpy as np
import itertools
from scipy.sparse import lil_matrix, linalg as splinalg

def get_states(N_total):
    """
    Genera todos los estados posibles para el sistema de dos poblaciones.
    Un estado es una tupla (s1, i1, r1, s2, i2, r2).
    La suma de todos los compartimentos debe ser igual a N_total.
    """
    # Genera todas las combinaciones de 6 números que suman N_total
    # Esto es un problema clásico de "stars and bars"
    all_states = []
    for combo in itertools.combinations(range(N_total + 5), 5):
        s1 = combo[0]
        i1 = combo[1] - combo[0] - 1
        r1 = combo[2] - combo[1] - 1
        s2 = combo[3] - combo[2] - 1
        i2 = combo[4] - combo[3] - 1
        r2 = N_total - s1 - i1 - r1 - s2 - i2
        
        if all(x >= 0 for x in [s1, i1, r1, s2, i2, r2]):
             all_states.append((s1, i1, r1, s2, i2, r2))
             
    return all_states

def generate_transition_matrix(N_total, params):
    """
    Construye la matriz de transición (Hamiltoniano) para el modelo SIRS de metapoblaciones.

    Args:
        N_total (int): La población total fija en el sistema (N1 + N2).
        params (dict): Un diccionario con los parámetros del modelo:
                       beta1, gamma1, xi1, beta2, gamma2, xi2,
                       mu_S, mu_I, mu_R.

    Returns:
        scipy.sparse.csr_matrix: La matriz de transición en formato disperso.
        list: La lista de estados correspondientes a los índices de la matriz.
        dict: Un mapeo de tuplas de estado a sus índices en la matriz.
    """
    states = get_states(N_total)
    num_states = len(states)
    state_to_idx = {state: i for i, state in enumerate(states)}

    # Usamos una matriz lil para una construcción eficiente
    H = lil_matrix((num_states, num_states), dtype=float)

    for i, state in enumerate(states):
        s1, i1, r1, s2, i2, r2 = state
        N1 = s1 + i1 + r1
        N2 = s2 + i2 + r2

        total_rate_out = 0.0

        # --- Reacciones en la Población 1 ---
        # Infección: S1 -> I1
        if s1 > 0 and i1 > 0:
            rate = (params['beta1'] / N1) * s1 * i1 if N1 > 0 else 0
            if rate > 0:
                next_state = (s1 - 1, i1 + 1, r1, s2, i2, r2)
                j = state_to_idx[next_state]
                H[j, i] = rate
                total_rate_out += rate
        
        # Recuperación: I1 -> R1
        if i1 > 0:
            rate = params['gamma1'] * i1
            next_state = (s1, i1 - 1, r1 + 1, s2, i2, r2)
            j = state_to_idx[next_state]
            H[j, i] = rate
            total_rate_out += rate

        # Pérdida de inmunidad: R1 -> S1
        if r1 > 0:
            rate = params['xi1'] * r1
            next_state = (s1 + 1, i1, r1 - 1, s2, i2, r2)
            j = state_to_idx[next_state]
            H[j, i] = rate
            total_rate_out += rate

        # --- Reacciones en la Población 2 ---
        # Infección: S2 -> I2
        if s2 > 0 and i2 > 0:
            rate = (params['beta2'] / N2) * s2 * i2 if N2 > 0 else 0
            if rate > 0:
                next_state = (s1, i1, r1, s2 - 1, i2 + 1, r2)
                j = state_to_idx[next_state]
                H[j, i] = rate
                total_rate_out += rate

        # Recuperación: I2 -> R2
        if i2 > 0:
            rate = params['gamma2'] * i2
            next_state = (s1, i1, r1, s2, i2 - 1, r2 + 1)
            j = state_to_idx[next_state]
            H[j, i] = rate
            total_rate_out += rate

        # Pérdida de inmunidad: R2 -> S2
        if r2 > 0:
            rate = params['xi2'] * r2
            next_state = (s1, i1, r1, s2 + 1, i2, r2 - 1)
            j = state_to_idx[next_state]
            H[j, i] = rate
            total_rate_out += rate
        
        # --- Migración entre poblaciones ---
        # S1 -> S2
        if s1 > 0:
            rate = params['mu_S'] * s1
            next_state = (s1 - 1, i1, r1, s2 + 1, i2, r2)
            j = state_to_idx[next_state]
            H[j, i] = rate
            total_rate_out += rate

        # S2 -> S1
        if s2 > 0:
            rate = params['mu_S'] * s2
            next_state = (s1 + 1, i1, r1, s2 - 1, i2, r2)
            j = state_to_idx[next_state]
            H[j, i] = rate
            total_rate_out += rate
        
        # I1 -> I2
        if i1 > 0:
            rate = params['mu_I'] * i1
            next_state = (s1, i1 - 1, r1, s2, i2 + 1, r2)
            j = state_to_idx[next_state]
            H[j, i] = rate
            total_rate_out += rate

        # I2 -> I1
        if i2 > 0:
            rate = params['mu_I'] * i2
            next_state = (s1, i1 + 1, r1, s2, i2 - 1, r2)
            j = state_to_idx[next_state]
            H[j, i] = rate
            total_rate_out += rate

        # R1 -> R2
        if r1 > 0:
            rate = params['mu_R'] * r1
            next_state = (s1, i1, r1 - 1, s2, i2, r2 + 1)
            j = state_to_idx[next_state]
            H[j, i] = rate
            total_rate_out += rate

        # R2 -> R1
        if r2 > 0:
            rate = params['mu_R'] * r2
            next_state = (s1, i1, r1 + 1, s2, i2, r2 - 1)
            j = state_to_idx[next_state]
            H[j, i] = rate
            total_rate_out += rate
            
        # El elemento diagonal es el negativo de la suma de todas las tasas de salida
        H[i, i] = -total_rate_out

    # Convertir a formato CSR para operaciones matriciales eficientes
    return H.tocsr(), states, state_to_idx

def evolve_state_vector(H, initial_vector, t):
    """
    Evoluciona el vector de estado en el tiempo usando la exponencial de la matriz.

    Calcula P(t) = expm(H*t) @ P(0)

    Args:
        H (scipy.sparse.spmatrix): El Hamiltoniano (matriz de transición).
        initial_vector (np.ndarray): El vector de estado inicial P(0).
        t (float): El tiempo hasta el cual evolucionar el sistema.

    Returns:
        np.ndarray: El vector de estado evolucionado P(t).
    """
    # expm_multiply es una forma eficiente de calcular la acción de la matriz exponencial
    # en un vector, sin calcular la matriz exponencial completa.
    return splinalg.expm_multiply(H * t, initial_vector)
