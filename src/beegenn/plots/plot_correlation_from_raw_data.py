import numpy as np
from pathlib import Path
import pandas as pd
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from .data_manager import DataManager
import scipy.cluster.hierarchy as sch
from matplotlib.colors import ListedColormap
import os, psutil
process = psutil.Process()
import matplotlib as mpl
import sys

def plot_correlation_per_pop(correlation, pop, subplot):
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    res = sns.heatmap(correlation, mask = mask, cmap = 'plasma', ax = subplot, cbar = False, xticklabels=True, yticklabels=True, vmin = -0.34, vmax = 0.53)
    subplot.set_title(pop, fontsize = 35)
    subplot.set_xlabel("Glomeruli", fontsize = 30)
    subplot.set_ylabel("Glomeruli", fontsize = 30)
    return res

def get_subplots(n_pops):
    figure, subplots = plt.subplots(
            1,
            n_pops + 1,
            gridspec_kw=dict(width_ratios=[1 for _ in range(n_pops)] + [0.1]),
            figsize = (n_pops*10, 10)
            )
    return figure, subplots

def colorbar(image, subplots, figure):
    p = [subplot.get_position().get_points().flatten() for subplot in subplots]
    ax_cbar = figure.add_axes([p[0][0], 0, p[-1][0], 0.05])
    cbar = plt.colorbar(image, cax=ax_cbar, orientation='horizontal')
    cbar.ax.set_xlabel("Correlation")

def get_cmap():
    n_colors = 256
    alpha_levels = [max(min(i/n_colors, 0.8), 0) for i in range(n_colors)]
    colors = [(1, 1, 1, alpha) for alpha in alpha_levels]
    cmap = ListedColormap(colors)
    return cmap



def plot_correlation_heatmap(pops, t_start, t_end, root_dir):

    figure, subplots = get_subplots(len(pops))

    for (pop, subplot) in zip(pops, subplots):
        correlation_matrix = pd.read_csv(root_dir / 'raw_data' / '0' / 'correlation_clustered' / f"{pop}_{t_start:.1f}_{t_end:.1f}.csv", index_col = 0)
        print(correlation_matrix)
        plot_correlation_per_pop(
            correlation_matrix,
            pop,
            subplot,
            )

    cbar = figure.colorbar(subplots[-2].collections[0], cax = subplots[-1])
    cbar.ax.tick_params(labelsize = 20)
    figure.tight_layout()
    plt.savefig(str(root_dir / '..' / 'correlation_final' / root_dir.name) + '.png', dpi = 200)
    plt.cla()
    plt.clf()


if __name__ == "__main__":
    sns.set(font_scale = 0.4)
    root_dir = Path(sys.argv[1])

    plot_correlation_heatmap(['orn', 'ln', 'pn'], 0.0, 60000.0, root_dir)
