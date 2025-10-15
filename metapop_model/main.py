
import os
import numpy as np
import scipy.sparse as sp
from model import generate_transition_matrix
from simulation import run_simulation # Ya no se importa create_delta_t_matrix
from plotting import create_heatmap_video

def main():
    """
    Script principal para simular y visualizar la dinámica de la metapoblación.
    """
    # --- Parámetros de Simulación ---
    # Ahora puedes intentar con valores más grandes de N_total gracias a la optimización de memoria.
    N_total = 8      # Población total
    delta_t = 0.5    # Paso de tiempo para la simulación discreta
    num_steps = 100  # Número de pasos a simular

    # --- Condición Inicial de la Simulación ---
    initial_state_tuple = (N_total - 1, 1, 0, 0, 0, 0)
    
    # --- Parámetros del Modelo ---
    params = {
        'beta1': 0.5,
        'gamma1': 0.1,
        'xi1': 0.05,
        'beta2': 0.2,
        'gamma2': 0.1,
        'xi2': 0.05,
        'mu_S': 0.1,
        'mu_I': 0.2,
        'mu_R': 0.1
    }

    # --- Archivos para guardar/cargar la matriz H y el mapa de estados ---
    h_matrix_file = os.path.join("metapop_model", f"H_N{N_total}.npz")
    states_file = os.path.join("metapop_model", f"states_N{N_total}.npy")
    state_to_idx_file = os.path.join("metapop_model", f"state_to_idx_N{N_total}.npy")

    # --- Paso 1: Generar o Cargar el Hamiltoniano y su mapa de estados ---
    if os.path.exists(h_matrix_file) and os.path.exists(states_file):
        print(f"--- Paso 1: Cargando Hamiltoniano y estados pre-calculados para N={N_total} ---")
        H = sp.load_npz(h_matrix_file)
        states = np.load(states_file, allow_pickle=True)
        state_to_idx = np.load(state_to_idx_file, allow_pickle=True).item()
        print(f"Modelo cargado. Hamiltoniano (H) tiene dimensiones: {H.shape}")

    else:
        print(f"--- Paso 1: Generando Hamiltoniano y mapa de estados para N={N_total} ---")
        H, states_list, state_to_idx = generate_transition_matrix(N_total, params)
        states = np.array(states_list, dtype=object)
        print(f"Modelo generado. Hamiltoniano (H) tiene dimensiones: {H.shape}")
        
        print("Guardando el modelo (H y mapa de estados) para uso futuro...")
        sp.save_npz(h_matrix_file, H)
        np.save(states_file, states)
        np.save(state_to_idx_file, state_to_idx)

    # --- Paso 2 (eliminado): Ya no se crea la matriz P_delta_t completa ---
    # print("\n--- Paso 2: Creando Matriz de Transición P(delta_t) ---")
    # P_delta_t = create_delta_t_matrix(H, delta_t)

    # --- Paso 3: Ejecutar Simulación (ahora es el paso 2) ---
    print("\n--- Paso 2: Ejecutando la Simulación (con optimización de memoria) ---")
    try:
        if sum(initial_state_tuple) != N_total:
            raise ValueError(f"La suma de los componentes de initial_state_tuple debe ser {N_total}")

        initial_idx = state_to_idx[initial_state_tuple]
        initial_vector = np.zeros(len(states))
        initial_vector[initial_idx] = 1.0

        print(f"Condición inicial: {initial_state_tuple}")
        print(f"Simulando por {num_steps} pasos con delta_t = {delta_t}...")
        # La función de simulación ahora necesita H y delta_t directamente
        history = run_simulation(H, initial_vector, delta_t, num_steps)
        print("Simulación completada.")

        # --- Paso 4: Generar Video (ahora es el paso 3) ---
        print("\n--- Paso 3: Generando Video de la Simulación ---")
        output_filename = os.path.join("metapop_model", "heatmap_animation.mp4")
        create_heatmap_video(history, list(states), N_total, delta_t, filename=output_filename)

    except KeyError:
        print(f"Error: La condición inicial {initial_state_tuple} no existe en el mapa de estados.")
    except Exception as e:
        print(f"Ocurrió un error inesperado durante la ejecución: {e}")

if __name__ == "__main__":
    main()
