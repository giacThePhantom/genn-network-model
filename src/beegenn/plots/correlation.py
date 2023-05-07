import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib
import matplotlib.pyplot as plt
from .data_manager import DataManager


def plot_correlation_per_pop(correlation, pop, subplot):
    res = subplot.matshow(correlation, cmap = 'plasma', aspect = 'equal')
    subplot.set_xticklabels([int(i) for i in subplot.get_xticks()], rotation=45, fontsize = 7)
    subplot.set_yticklabels([int(i) for i in subplot.get_yticks()], rotation=45, fontsize = 7)
    subplot.set_title(pop)
    subplot.set_xlabel("Glomeruli")
    subplot.set_ylabel("Glomeruli")
    return res

def plot_sdf_over_time_outliers(sdf_matrix_avg, subplot):
    glomeruli_mean_sdf = np.mean(sdf_matrix_avg, axis = 1)
    global_mean = np.mean(glomeruli_mean_sdf)
    global_sdt = np.std(glomeruli_mean_sdf)
    glomeruli_of_interest = []

    for (i, mean_sdf) in enumerate(glomeruli_mean_sdf):
        if np.abs(mean_sdf - global_mean) > global_sdt:
            glomeruli_of_interest.append(i)

    for i in glomeruli_of_interest:
        subplot.plot(sdf_matrix_avg[i, :], label = f"Glomerulus {i}")
    subplot.legend()

def get_subplots(n_pops):
    figure = plt.figure()
    subplots = ImageGrid(figure, 111,
                    nrows_ncols = (1,n_pops),
                    axes_pad = 0.05,
                    cbar_location = "right",
                    cbar_mode="single",
                    cbar_size="5%",
                    cbar_pad=0.05
                    )
    return figure, subplots

def colorbar(image, subplots):
    cbar = plt.colorbar(image, cax=subplots.cbar_axes[0])
    cbar.ax.set_ylabel("Correlation")

def plot_correlation_heatmap(pops, t_start, t_end, data_manager, show):


    figure, subplots = get_subplots(len(pops))
    image = []

    for (pop, subplot) in zip(pops, subplots.axes_all):
        sdf_avg = data_manager.sdf_per_glomerulus_avg(
                pop,
                t_start,
                t_end
                )
        correlation_matrix = data_manager.sdf_correlation(sdf_avg)
        image.append(
                plot_correlation_per_pop(
                    correlation_matrix,
                    pop,
                    subplot
                    )
                )
    colorbar(image[-1], subplots)
    filename = f"correlation/{t_start:.1f}_{t_end:.1f}.png"
    data_manager.show_or_save(filename, show)

if __name__ == "__main__":
    from beegenn.parameters.reading_parameters import parse_cli
    from pathlib import Path
    import pandas as pd
    param = parse_cli()
    data_manager = DataManager(param['simulations']['simulation'], param['simulations']
                 ['name'], param['neuron_populations'], param['synapses'])

    events = pd.read_csv(Path(param['simulations']['simulation']['output_path']) / param['simulations']['name'] / 'events.csv')
    print(events)

    for i, row in events.iterrows():
        plot_correlation_heatmap(['orn', 'pn', 'ln'], row['t_start'], row['t_end'], data_manager, show = False)
