
from .model import generate_transition_matrix
import numpy as np

def main():
    """
    Script principal para probar el generador de la matriz de transición.
    """
    # --- Parámetros del Modelo ---
    # ADVERTENCIA: N_total > 8 puede ser computacionalmente muy intensivo.
    # Para N_total = 200, la matriz es demasiado grande para ser generada.
    N_total = 4

    # Estado inicial: (s1, i1, r1, s2, i2, r2)
    # Por ejemplo, 2 susceptibles en el nodo 1 y 2 susceptibles en el nodo 2.
    initial_state_vector = np.zeros(1) # El tamaño se ajustará después
    
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
    
    # --- Prueba con un vector de población inicial ---
    # Se necesita un vector que represente la probabilidad de estar en cada estado.
    # Para una simulación, se suele empezar con una certeza del 100% en un estado inicial.
    
    # Ejemplo: Empezar con el estado (s1=2, i1=0, r1=0, s2=2, i2=0, r2=0)
    try:
        start_state = (1, 1, 0, 2, 0, 0)
        start_idx = state_to_idx[start_state]
        
        # Vector de estado inicial (un 1 en la posición del estado inicial)
        initial_state_vector = np.zeros(len(states))
        initial_state_vector[start_idx] = 1.0

        print(f"\nProbando la aplicación del Hamiltoniano al vector del estado inicial {start_state}")

        # Aplicar el Hamiltoniano al vector de estado
        # Esto calcula la derivada temporal del vector de probabilidad, dP/dt = H * P
        dP_dt = H.dot(initial_state_vector)

        print("Vector de estado inicial (P):")
        print(initial_state_vector)

        print("\nResultado de H * P (representa dP/dt en t=0):")
        # Mostramos solo los elementos no nulos para claridad
        non_zero_indices = dP_dt.nonzero()[0]
        for i in non_zero_indices:
            print(f"  -> Estado {states[i]}: Tasa de cambio = {dP_dt[i]}")

    except KeyError:
        print(f"\nError: El estado inicial (2,0,0,2,0,0) no es válido para N_total = {N_total}.")
    except Exception as e:
        print(f"\nOcurrió un error: {e}")

if __name__ == "__main__":
    main()
