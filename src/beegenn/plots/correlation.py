import numpy as np
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

mpl.use('agg')


def cluster(correlation):
    correlation[np.isnan(correlation)] = -1
    correlation = pd.DataFrame(correlation, columns = np.arange(0, correlation.shape[0]))
    pairwise_distances = sch.distance.pdist(correlation)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold,
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)

    if isinstance(correlation, pd.DataFrame):
        return correlation.iloc[idx, :].T.iloc[idx, :]

    return correlation[idx, :][:, idx]

def plot_correlation_per_pop(correlation, to_cluster, to_mask, mask, pop, subplot):
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    if to_cluster:
        correlation_df = cluster(correlation)
    else:
        correlation_df = pd.DataFrame(correlation, columns = np.arange(0, correlation.shape[0]))
    res = sns.heatmap(correlation_df, mask = mask, cmap = 'plasma', ax = subplot, cbar = False, xticklabels=True, yticklabels=True, vmin = -1, vmax = 1)
    if to_mask:
        res = sns.heatmap(mask, cmap = get_cmap(), ax = subplot, cbar = False, xticklabels=True, yticklabels=True, vmin = -1, vmax = 1)
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



def plot_correlation_heatmap(pops, t_start, t_end, data_manager, nrun, to_cluster = False, to_mask = False, show = False):

    figure, subplots = get_subplots(len(pops))

    for (pop, subplot) in zip(pops, subplots):
        sdf_avg = data_manager.sdf_per_glomerulus_avg(
                pop,
                t_start,
                t_end,
                str(nrun)
                )
        correlation_matrix = data_manager.sdf_correlation(sdf_avg)
        mask = np.ones_like(correlation_matrix, dtype=bool)
        if to_mask:
            glomeruli_of_interest = data_manager.get_active_glomeruli_per_pop(
                    sdf_avg
                    )
            for i in glomeruli_of_interest:
                for j in glomeruli_of_interest:
                    mask[i,j] = False
        plot_correlation_per_pop(
            correlation_matrix,
            to_cluster,
            to_mask,
            mask,
            pop,
            subplot
            )

        data_manager.save_raw_data(
                f"correlation{('_not' if not to_cluster else '')}_clustered/{pop}_{t_start:.1f}_{t_end:.1f}.csv",
                correlation_matrix,
                str(nrun)
                )
    figure.colorbar(subplots[-2].collections[0], cax = subplots[-1])
    figure.tight_layout()
    cluster_name = ("_not" if not cluster else "") + "_clustered"
    masked_name = ("_not" if not to_mask else "") + "_masked"
    filename = f"correlation{cluster_name}{masked_name}/{t_start:.1f}_{t_end:.1f}.png"
    data_manager.show_or_save(filename, str(nrun), show)


if __name__ == "__main__":
    sns.set(font_scale = 0.4)
    from beegenn.parameters.reading_parameters import parse_cli
    from pathlib import Path
    import pandas as pd
    param = parse_cli()
    data_manager = DataManager(param['simulations']['simulation'], param['simulations']
                 ['name'], param['neuron_populations'], param['synapses'])

    events = pd.read_csv(Path(param['simulations']['simulation']['output_path']) / param['simulations']['name'] / 'events.csv')

    if len(events.index) > 0:
        for i, row in events.iterrows():
            for run in range(data_manager.get_nruns()):
                plot_correlation_heatmap(['orn', 'pn', 'ln'], row['t_start'], row['t_end'], data_manager, run, to_cluster = False, to_mask = False, show = False)
                plot_correlation_heatmap(['orn', 'pn', 'ln'], row['t_start'], row['t_end'], data_manager, run, to_cluster = True, to_mask = False, show = False)

    else:
        for t_start in range(60000, int(data_manager.protocol.simulation_time), 120000):
            t_end = t_start + 60000
            for run in range(data_manager.get_nruns()):
                plot_correlation_heatmap(['orn', 'pn', 'ln'], t_start, t_end, data_manager, run, to_cluster = False, to_mask = False, show = False)
                plot_correlation_heatmap(['orn', 'pn', 'ln'], t_start, t_end, data_manager, run, to_cluster = True, to_mask = False, show = False)
