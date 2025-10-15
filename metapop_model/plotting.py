
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

def aggregate_probabilities(history, states, N_total):
    """
    Agrega las probabilidades de los microestados en macroestados (Total S, Total I).
    """
    num_steps = len(history)
    # El tamaño es N+1 porque podemos tener de 0 a N individuos
    aggregated_data = np.zeros((num_steps, N_total + 1, N_total + 1))

    for t, state_vector in enumerate(history):
        for i, prob in enumerate(state_vector):
            if prob > 0:
                s1, i1, _, s2, i2, _ = states[i]
                total_s = s1 + s2
                total_i = i1 + i2
                aggregated_data[t, total_s, total_i] += prob
                
    return aggregated_data

def create_heatmap_video(history, states, N_total, delta_t, filename="epidemic_heatmap.mp4"):
    """
    Crea un video de heatmaps que muestra la evolución de las probabilidades P(S_total, I_total).

    Args:
        history (list): El historial de vectores de estado de la simulación.
        states (list): La lista de todos los microestados.
        N_total (int): La población total.
        delta_t (float): El paso de tiempo, para etiquetar los fotogramas.
        filename (str): El nombre del archivo de video de salida.
    """
    print("Agregando probabilidades para la visualización...")
    aggregated_data = aggregate_probabilities(history, states, N_total)

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Usamos el primer fotograma para establecer los límites del color
    vmax = np.max(aggregated_data)
    if vmax == 0:
        print("No hay datos para graficar. El video no será generado.")
        return None
        
    cax = ax.imshow(aggregated_data[0].T, cmap='viridis', interpolation='nearest', origin='lower',
                    vmin=0, vmax=vmax)
    fig.colorbar(cax, label='Probabilidad')

    ax.set_xlabel("Número Total de Susceptibles (S_total)")
    ax.set_ylabel("Número Total de Infectados (I_total)")
    ax.set_title(f"Distribución de Probabilidad P(S_total, I_total) en t=0.0")
    ax.set_xticks(np.arange(N_total + 1))
    ax.set_yticks(np.arange(N_total + 1))

    def animate(i):
        # Transponemos los datos para que S esté en el eje x e I en el eje y
        cax.set_data(aggregated_data[i].T)
        ax.set_title(f"Distribución de Probabilidad P(S_total, I_total) en t={i * delta_t:.2f}")
        return [cax]

    print(f"Generando animación... Esto puede tardar un momento.")
    # Intervalo en milisegundos entre fotogramas
    anim = animation.FuncAnimation(fig, animate, frames=len(history), interval=100, blit=True)
    
    # Guardar el video
    try:
        anim.save(filename, writer='ffmpeg', fps=10, dpi=100)
        print(f"Video guardado como '{filename}'")
        plt.close(fig) # Cerrar la figura para no mostrarla en entornos interactivos
        # Para mostrar en un notebook, se podría devolver HTML(anim.to_jshtml())
        return filename
    except Exception as e:
        print(f"Error al guardar el video: {e}")
        print("Asegúrate de que ffmpeg esté instalado y en el PATH de tu sistema.")
        plt.close(fig)
        return None
