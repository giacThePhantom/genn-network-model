import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from .data_manager import DataManager

mpl.use('agg')


def plot_sdf_heatmap_per_pop(sdf_average, t_start, t_end, dt, pop, subplot):
    res = subplot.imshow(sdf_average, vmin = 0, cmap = 'plasma')
    subplot.set_aspect((t_end-t_start)//10)
    subplot.set_title(pop)
    subplot.set_xlabel("Time [ms]")
    subplot.set_ylabel("Glomeruli")
    return res


def plot_sdf_over_time_outliers(sdf_matrix_avg, subplot):
    glomeruli_mean_sdf = np.mean(sdf_matrix_avg, axis=1)
    global_mean = np.mean(glomeruli_mean_sdf)
    global_sdt = np.std(glomeruli_mean_sdf)
    glomeruli_of_interest = []

    for i, mean_sdf in enumerate(glomeruli_mean_sdf):
        if np.abs(mean_sdf - global_mean) > global_sdt:
            glomeruli_of_interest.append(i)

    for i in glomeruli_of_interest:
        subplot.plot(sdf_matrix_avg[i, :], label=f"Glomerulus {i}")
    subplot.legend()


def get_subplots(n_pops):
    figure, subplots = plt.subplots(
        1, n_pops, sharey=True, layout="constrained")
    if n_pops == 1:
        subplots = [subplots]
    return figure, subplots


def colorbar(image, subplot, figure):
    cbar = figure.colorbar(image, ax=subplot)
    cbar.ax.set_ylabel("SDF ($Hz$)")


def plot_sdf_heatmap(pops, t_start, t_end, data_manager, run, show):
    figure, subplots = get_subplots(len(pops))
    image = []

    for (pop, subplot) in zip(pops, subplots):
        sdf_avg = data_manager.sdf_per_glomerulus_avg(
                pop,
                t_start,
                t_end,
                run
                )
        raw_sdf_filename = data_manager._root_raw_data_dir / str(run) / "sdf" / f"{pop}_{t_start:.1f}_{t_end:.1f}.csv"
        raw_sdf_filename.parent.mkdir(parents=True, exist_ok=True)
        df_col_name = (np.arange(0, sdf_avg.shape[1]) * data_manager.get_sim_dt()) + t_start
        df = pd.DataFrame(sdf_avg,
                          columns = df_col_name,
                          index = np.arange(0, sdf_avg.shape[0]) + 1
                          )
        df.T.to_csv(raw_sdf_filename)
        image.append(
            plot_sdf_heatmap_per_pop(
                sdf_avg, t_start, t_end, data_manager.get_sim_dt(), pop, subplot
            )
        )
    colorbar(image[-1], subplots[-1], figure)
    filename = f"sdf/{t_start:.1f}_{t_end:.1f}.png"
    data_manager.show_or_save(filename, run, show)


if __name__ == "__main__":
    from beegenn.parameters.reading_parameters import parse_cli
    from pathlib import Path
    import pandas as pd

    param = parse_cli()
    data_manager = DataManager(
        param["simulations"]["simulation"],
        param["simulations"]["name"],
        param["neuron_populations"],
        param["synapses"],
    )

    events = pd.read_csv(Path(param['simulations']['simulation']['output_path']) / param['simulations']['name'] / 'events.csv')

    if len(events.index) > 0:
        for i, row in events.iterrows():
            for i in range(data_manager.get_nruns()):
                plot_sdf_heatmap(['orn', 'ln', 'pn'], row['t_start'], row['t_start'] + 3000 , data_manager, str(i), show = False)

            plot_sdf_heatmap(['orn', 'ln', 'pn'], row['t_start'], row['t_start'] + 3000, data_manager, 'mean', show = False)

    else:
        for t_start in range(00000, int(data_manager.protocol.simulation_time), 120000):
            t_end = t_start + 3000
            for i in range(data_manager.get_nruns()):
                plot_sdf_heatmap(['orn', 'ln', 'pn'], t_start, t_end, data_manager, str(i), show = False)

            plot_sdf_heatmap(['orn', 'ln', 'pn'], t_start, t_end, data_manager, 'mean', show = False)
