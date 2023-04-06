import numpy as np
import matplotlib.pyplot as plt
from .data_manager import DataManager

def plot_connectivity_per_pop(connectivity_matrix, first_repeat, second_repeat, title, subplot):
    matrix = np.repeat(
            connectivity_matrix,
            repeats = first_repeat//connectivity_matrix.shape[0],
            axis = 0
            )
    matrix = np.repeat(
            matrix,
            repeats = second_repeat//connectivity_matrix.shape[1],
            axis = 1
            )
    res = subplot.imshow(matrix, cmap = 'plasma')
    subplot.set_title(title)
    return res


def get_subplots(n_synapses):
    figure, subplot = plt.subplots(1, n_synapses + 1, layout = 'constrained')
    return figure, subplot

def plot_inhibitory_connectivity(synapses, data_manager, show):
    figure, subplots = get_subplots(len(synapses))

    connectivity_matrix = data_manager.get_inhibitory_connectivity()


    plot_connectivity_per_pop(
            connectivity_matrix,
            data_manager.get_pop_size('or'),
            data_manager.get_pop_size('or'),
            "Standard",
            subplots[0]
            )


    for ((source, target), subplot) in zip(synapses, subplots[1:]):
        plot_connectivity_per_pop(
                connectivity_matrix,
                data_manager.get_pop_size(source),
                data_manager.get_pop_size(target),
                f"{source} to {target}",
                subplot,
                )

    filename = "connectivity/inhibitory_connectivity.png"
    data_manager.show_or_save(filename, show)

if __name__ == "__main__":
    from beegenn.parameters.reading_parameters import parse_cli
    from pathlib import Path
    import pandas as pd
    param = parse_cli()
    data_manager = DataManager(param['simulations']['simulation'], param['simulations']
                 ['name'], param['neuron_populations'], param['synapses'])

    events = pd.read_csv(Path(param['simulations']['simulation']['output_path']) / param['simulations']['name'] / 'events.csv')

    plot_inhibitory_connectivity([('ln', 'pn'), ('ln', 'ln')], data_manager, show = False)
