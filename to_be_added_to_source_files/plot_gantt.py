"""
TODO: move under plots
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from beegenn.parameters.reading_parameters import parse_cli


def make_gantt(params, dataframe):
    """
    Make a Gantt plot of the protocol events. It will be saved to

    Arguments
    ========
    params:
        the parameters (loaded by param_cli or similar methods)
    dataframe: pd.DataFrame
        The events dataframe (loaded from events.csv)
    """
    dataframe = dataframe[dataframe['concentration'] > 0]
    dataframe['duration'] = dataframe['t_end'] - dataframe['t_start']
    t_min = int(dataframe['t_start'].min())
    t_max = int(dataframe['t_end'].max())
    tick_step = 3000


    # This transforms concentrations into colours.
    # It works by log-norming of the concentration
    # so that it stays between 0 and 1, and applying a colormap that
    # turns the float into a useful color
    min_concentration = 1e-7
    max_concentration = 1.0
    norm = colors.LogNorm(min_concentration, max_concentration)

    #cmap = plt.get_cmap('plasma')
    #dataframe['color'] = dataframe['concentration'].apply(lambda x: cmap(norm(x)))
    dataframe['color'] = dataframe['concentration'].apply(lambda x: [0, 0, 1, norm(x)])

    fig, ax = plt.subplots(1, figsize=(16, 2))

    ax.barh(dataframe['odor_name'], dataframe.duration, left=dataframe.t_start, color=dataframe.color)

    my_colors = np.array([[0, 0, 1, 0.0], [0, 0, 1, 1.0]])

    cmap = colors.LinearSegmentedColormap.from_list('my_color', my_colors, N=2)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)

    ticks = np.arange(t_min, t_max, tick_step)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks // 1000)
    out_dir = params['simulations']['simulation']['output_path']
    plt.savefig(Path(out_dir) / name / "gantt.png")


if __name__ == "__main__":
    params = parse_cli()
    name = params['simulations']['name']
    out_dir = params['simulations']['simulation']['output_path']
    df = pd.read_csv(Path(out_dir) / name / "events.csv")
    make_gantt(params, df)
