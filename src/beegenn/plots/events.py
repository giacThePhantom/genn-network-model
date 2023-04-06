import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np
from .data_manager import DataManager


def get_subplots():
    figure, subplots = plt.subplots(1, figsize=(16, 2))
    return figure, subplots

def assign_color_to_concentration(events, min_concentration, max_concentration):
    # This transforms concentrations into colours.
    # It works by log-norming of the concentration
    # so that it stays between 0 and 1, and applying a colormap that
    # turns the float into a useful color
    norm = colors.LogNorm(min_concentration, max_concentration)
    events['color'] = events['concentration'].apply(lambda x: [0, 0, 1, norm(x)])
    return norm

def create_colormap(norm):
    my_colors = np.array(
            [
                [0, 0, 1, 0.0],
                [0, 0, 1, 1.0]
                ]
            )
    cmap = colors.LinearSegmentedColormap.from_list('my_color', my_colors, N=2)
    cm.ScalarMappable(norm=norm, cmap=cmap)

def add_ticks(t_min, t_max, tick_step, subplots):
    ticks = np.arange(t_min, t_max, tick_step)
    subplots.set_xticks(ticks)
    subplots.set_xticklabels(ticks // 1000)

def plot_gantt(data_manager, show):
    """
    Make a Gantt plot of the protocol events. It will be saved to

    Arguments
    ========
    params:
        the parameters (loaded by param_cli or similar methods)
    dataframe: pd.DataFrame
        The events dataframe (loaded from events.csv)
    """
    events = data_manager.get_events()
    events = events[events['concentration'] > 0]
    events.loc[:,'duration'] = events.loc[:,'t_end'] - events.loc[:,'t_start']
    t_min = int(events['t_start'].min())
    t_max = int(events['t_end'].max()) + 1
    tick_step = int(np.mean(events.loc[:,'t_end']-events.loc[:,'t_start']))
    # norm = assign_color_to_concentration(events, 1e-7, 1)
    norm = assign_color_to_concentration(events, min(events['concentration']), max(events['concentration']))

    figure, subplots = get_subplots()
    subplots.barh(
            events['odor_name'],
            events.duration,
            left=events.t_start,
            color=events.color
            )
    create_colormap(norm)
    add_ticks(t_min, t_max, tick_step, subplots)

    filename = 'events/gantt.png'
    data_manager.show_or_save(filename, show)

if __name__ == "__main__":
    from beegenn.parameters.reading_parameters import parse_cli
    from pathlib import Path
    import pandas as pd
    param = parse_cli()
    data_manager = DataManager(param['simulations']['simulation'], param['simulations']
                 ['name'], param['neuron_populations'], param['synapses'])

    plot_gantt(data_manager, False)
    data_manager.close()
