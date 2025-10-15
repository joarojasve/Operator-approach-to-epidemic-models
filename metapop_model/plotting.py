
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
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
    Usa una escala logarítmica para el color para una mejor visualización.

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
    
    # --- Configuración de la escala logarítmica ---
    vmax = np.max(aggregated_data)
    # Encontrar la probabilidad mínima no nula para establecer el límite inferior
    non_zero_probs = aggregated_data[aggregated_data > 0]
    vmin = np.min(non_zero_probs) if len(non_zero_probs) > 0 else 0

    # Si no hay datos o la probabilidad es uniforme, no se puede usar escala log.
    if vmax == 0 or vmin >= vmax:
        print("Advertencia: No se pueden generar datos para el heatmap (probabilidades son cero o uniformes).")
        norm = None
        cbar_label = "Probabilidad (lineal)"
    else:
        # Usar LogNorm. clip=True mapea los valores 0 al color de vmin.
        norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
        cbar_label = "Probabilidad (escala log)"

    # --- Creación del primer fotograma ---
    cax = ax.imshow(aggregated_data[0].T, cmap='viridis', interpolation='nearest', origin='lower', norm=norm)
    fig.colorbar(cax, label=cbar_label)

    ax.set_xlabel("Número Total de Susceptibles (S_total)")
    ax.set_ylabel("Número Total de Infectados (I_total)")
    ax.set_xticks(np.arange(N_total + 1))
    ax.set_yticks(np.arange(N_total + 1))
    ax.set_title(f"Distribución de Probabilidad P(S_total, I_total) en t=0.0")

    def animate(i):
        cax.set_data(aggregated_data[i].T)
        ax.set_title(f"Distribución de Probabilidad P(S_total, I_total) en t={i * delta_t:.2f}")
        return [cax]

    print(f"Generando animación ({cbar_label})... Esto puede tardar un momento.")
    anim = animation.FuncAnimation(fig, animate, frames=len(history), interval=100, blit=True)
    
    try:
        anim.save(filename, writer='ffmpeg', fps=10, dpi=100)
        print(f"Video guardado como '{filename}'")
        plt.close(fig)
        return filename
    except Exception as e:
        print(f"Error al guardar el video: {e}")
        print("Asegúrate de que ffmpeg esté instalado y en el PATH de tu sistema.")
        plt.close(fig)
        return None
