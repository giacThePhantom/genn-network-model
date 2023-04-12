import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .data_manager import DataManager



def sdf_time_avg(sdf):
    return sdf.mean(axis = 1)


def amplitude_concentration_coefficient(events):
    res = events
    res['amplitude'] = pd.to_numeric(events['odor_name'].str.split(';').str[0].str.split(':').str[1])
    res['sigma'] = pd.to_numeric(events['odor_name'].str.split(';').str[1].str.split(':').str[1])
    res['coefficient'] = events['concentration']*np.power(10, events['amplitude'])
    return res

def split_by_coefficient(events):
    res = []

    for i in events['coefficient'].unique():
        res.append(events.loc[events['coefficient'] == i])
    return res

def get_active_glomeruli_per_var(events, data_manager, pop, var):
    active_glo = []
    var_values = []
    for (i, rows) in events.iterrows():
        sdf = data_manager.sdf_per_glomerulus_avg(pop, rows['t_start'], rows['t_end'])
        active_glo.append(data_manager.get_active_glomeruli_per_pop(sdf))
        var_values.append(rows[var])

    return active_glo, var_values

def get_subplots(n_pops):
    figure, subplots = plt.subplots(
            n_pops,
            sharey = True,
            layout = 'constrained'
            )

    if n_pops == 1:
        subplots = [subplots]
    return figure, subplots

def colorbar(image, subplot, figure):
    cbar = figure.colorbar(image, ax=subplot)
    cbar.ax.set_ylabel("Average SDF ($Hz$)")

def plot_active_glomeruli_over_amplitude(data_manager, pops, filters, explored_var, show):
    figure, subplots = get_subplots(len(pops))

    events = data_manager.get_events()
    for filter in filters:
        events = events.loc[events[filter] == filters[filter]]
    for (subplot, pop) in zip(subplots, pops):

        x = []
        y = []
        active_glo, amplitude = get_active_glomeruli_per_var(
            events,
            data_manager,
            pop,
            explored_var
            )
        for (glo, a) in zip(active_glo, amplitude):
            for i in glo:
                x.append(a)
                y.append(i)
        subplot = plt.scatter(x, y)

    data_manager.show_or_save("test", show)

    filename = f"odor_parameters/concentration_{events['concentration'][0]:.1f}.png"
    data_manager.show_or_save(filename, show)
    pass

if __name__ == "__main__":
    simulations = input().split()
    from beegenn.parameters.reading_parameters import parse_cli
    param = parse_cli()

    data_manager = DataManager(
            param['simulations']['simulation'],
            param['simulations']['name'],
            param['neuron_populations'],
            param['synapses']
            )
    amplitude_concentration_coefficient(data_manager.get_events())

    plot_active_glomeruli_over_amplitude(data_manager,
                                         [
                                             'orn'
                                             ],
                                         {
                                             'sigma' : 3.0
                                             },
                                         'coefficient',
                                         True)

    data_manager.close()
