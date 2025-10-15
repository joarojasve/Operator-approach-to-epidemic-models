
from .model import generate_transition_matrix, evolve_state_vector
import numpy as np

def main():
    """
    Script principal para probar el generador de la matriz de transición y la evolución temporal.
    """
    # --- Parámetros del Modelo ---
    # ADVERTENCIA: N_total > 8 puede ser computacionalmente muy intensivo.
    N_total = 4
    
    params = {
        'beta1': 0.3,   # Tasa de infección en la población 1
        'gamma1': 0.1,  # Tasa de recuperación en la población 1
        'xi1': 0.05,  # Tasa de pérdida de inmunidad en la población 1
        'beta2': 0.3,   # Tasa de infección en la población 2
        'gamma2': 0.1,  # Tasa de recuperación en la población 2
        'xi2': 0.05,  # Tasa de pérdida de inmunidad en la población 2
        'mu_S': 0.01, # Tasa de migración para susceptibles
        'mu_I': 0.02, # Tasa de migración para infectados
        'mu_R': 0.01  # Tasa de migración para recuperados
    }

    print(f"Generando la matriz de transición para N_total = {N_total}...")
    H, states, state_to_idx = generate_transition_matrix(N_total, params)
    print(f"La matriz de transición (Hamiltoniano) tiene dimensiones: {H.shape}")
    print(f"Número total de estados posibles: {len(states)}")

    # --- Prueba de Evolución Temporal ---
    # Estado inicial: Introducimos una infección en el nodo 1.
    # Empezamos en el estado (s1=3, i1=1, r1=0, s2=0, i2=0, r2=0)
    try:
        start_state = (3, 1, 0, 0, 0, 0)
        if N_total != sum(start_state):
             # Ajustar el estado inicial si N_total cambia
             start_state = (N_total - 1, 1, 0, 0, 0, 0) 

        start_idx = state_to_idx[start_state]
        initial_state_vector = np.zeros(len(states))
        initial_state_vector[start_idx] = 1.0

        print(f"\nEstado inicial: {start_state} (Probabilidad = 1.0)")

        # Tiempo de evolución
        t = 10.0
        print(f"Evolucionando el sistema hasta t = {t}...")

        # Calcular el vector de estado en el tiempo t
        final_state_vector = evolve_state_vector(H, initial_state_vector, t)

        print(f"\nDistribución de probabilidad en t = {t}:")
        # Mostramos los 5 estados más probables para mayor claridad
        # Se ordenan los índices por la probabilidad descendente
        most_probable_indices = np.argsort(final_state_vector)[::-1]

        total_prob = 0
        for i in range(min(5, len(states))):
            idx = most_probable_indices[i]
            prob = final_state_vector[idx]
            if prob > 1e-6: # Solo mostrar si la probabilidad no es despreciable
                print(f"  -> Estado {states[idx]}: Probabilidad = {prob:.4f}")
                total_prob += prob
        
        print(f"\nSuma de las probabilidades mostradas: {total_prob:.4f}")
        print(f"Suma total de probabilidades (verificación): {np.sum(final_state_vector):.4f}")

    except KeyError:
        print(f"\nError: El estado inicial {start_state} no es válido para N_total = {N_total}.")
    except Exception as e:
        print(f"\nOcurrió un error: {e}")

if __name__ == "__main__":
    main()
