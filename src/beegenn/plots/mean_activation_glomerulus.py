import matplotlib.pyplot as plt
from .data_manager import DataManager

def plot_active_glomeruli_sdf_per_pop(pop, t_start, t_end, data_manager, subplot):
    sdf_matrix_avg = data_manager.sdf_per_glomerulus_avg(
            pop,
            t_start,
            t_end,
            )
    glomeruli_of_interest = data_manager.get_active_glomeruli_per_pop(
            sdf_matrix_avg
            )
    for i in glomeruli_of_interest:
        subplot.plot(sdf_matrix_avg[i, :], label = f"{i}")
    subplot.legend()
    handles, labels = subplot.get_legend_handles_labels()
    subplot.legend().remove()
    return handles, labels

def get_subplots(n_pops):
    figure, subplots = plt.subplots(n_pops)
    return figure, subplots

def fix_labels(figure, handles, labels):
    fig_labels, fig_handles = [], []

    for (handle, label) in zip(handles, labels):
        for (h, l) in zip(handle, label):
            if l not in fig_labels:
                fig_labels.append(l)
                fig_handles.append(h)

    figure.legend(
            fig_handles,
            fig_labels,
            ncol = 6,
            title = "Glomeruli",
            loc = 'outside lower center',
            fancybox = True,
            shadow = True,
            bbox_to_anchor = (0., 1.02, 1., .102),
            borderaxespad = 0.0,
            )

def plot_outliers_sdf_over_time(pops, t_start, t_end, data_manager, show):
    figure, subplots = get_subplots(len(pops))

    handles = []
    labels = []
    for (pop, subplot) in zip(pops, subplots):
        handle, label = plot_active_glomeruli_sdf_per_pop(
                pop,
                t_start,
                t_end,
                data_manager,
                subplot,
                )
        handles.append(handle)
        labels.append(label)
    fix_labels(figure, handles, labels)


    filename = f"sdf_over_time/{t_start:.1f}_{t_end:.1f}.png"
    data_manager.show_or_save(filename, show)

if __name__ == "__main__":
    from beegenn.parameters.reading_parameters import parse_cli
    from pathlib import Path
    import pandas as pd
    param = parse_cli()
    data_manager = DataManager(param['simulations']['simulation'], param['simulations']
                 ['name'], param['neuron_populations'], param['synapses'])

    events = pd.read_csv(Path(param['simulations']['simulation']['output_path']) / param['simulations']['name'] / 'events.csv')


    for i, row in events.iterrows():
        plot_outliers_sdf_over_time(['orn', 'pn', 'ln'], row['t_start'], row['t_end'], data_manager, show = False)
