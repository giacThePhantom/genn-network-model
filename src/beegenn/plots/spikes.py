import numpy as np
import matplotlib.pyplot as plt
from .data_manager import DataManager

def rolling_average(data, kernel_dimension):
    kernel = np.ones((kernel_dimension,))/kernel_dimension
    convolved = np.convolve(data, kernel, mode='same')
    return convolved

def subplot_spike_pane(spike_t: np.ndarray, subplot):
    """
    Superimpose spikes on top of of an subplotis. A "spike" is a dirac-like impulse going
    from -70 to 20 mV. This function is necessary as Genn truncates the at the spike event *before* logging.

    Arguments
    ---------
    spike_t:
        The spike event times
    subplot:
        a Matplotlib subplotis
    """
    spike_t = np.reshape(spike_t, (1, -1))
    x = np.vstack((spike_t, spike_t))
    y = np.ones(x.shape)
    y[0, :] = 0.0
    y[1, :] = 10.0
    return x, y

def add_spikes_to_voltage(voltage, time, spike_times):
    spike_index = 0
    i = 0
    while i < len(time) and spike_index < len(spike_times):
        if time[i] == spike_times[spike_index]:
            voltage[i] = 20
            spike_index += 1
        if time[i] < spike_times[spike_index] and time[i+1] > spike_times[spike_index]:
            voltage = np.concatenate((voltage[:i], [20], voltage[i:]))
            time = np.concatenate((time[:i], [spike_times[spike_index]], time[i:]))
            spike_index += 1

        i += 1
    return voltage, time



def plot_voltage(voltage, time, spike_times, pop_name, id_neuron, subplot, kernel_dimension):
    subplot.set_title(f"Voltage for neuron {id_neuron} in {pop_name.upper()}")
    subplot.spines['right'].set_visible(False)
    subplot.spines['top'].set_visible(False)
    voltage_rolled_average = rolling_average(voltage, kernel_dimension)
    voltage_rolled_average, time = add_spikes_to_voltage(voltage_rolled_average, time, spike_times)
    subplot.plot(time, voltage_rolled_average, 'k', linewidth = 0.5)
    subplot.set_ylim([-90, 40])
    subplot.set_ylabel("mV")

def plot_spikes_for_population(spike_times, subplot):
    spike_times , spike_voltages = subplot_spike_pane(spike_times, subplot)
    subplot.plot(spike_times, spike_voltages, 'k', linewidth=0.4, color = 'blue')
    subplot.axes.yaxis.set_visible(False)


def or_most_active(ra):
    #Sum over time
    ra_sum = np.sum(ra, axis=0)
    #Pick the or that has been the most active during
    #The time window
    or_most_active = np.argmax(ra_sum)
    return or_most_active

def plot_ra(or_most_active, ra_times, ra, subplot):
    subplot.set_title(f"OR best ra activation rate (of neuron {or_most_active})")
    subplot.spines['right'].set_visible(False)
    subplot.spines['top'].set_visible(False)
    subplot.plot(ra_times, ra[:,or_most_active], 'k', linewidth=0.8)
    subplot.set_xlim([ra_times[0], ra_times[-1]])
    subplot.set_ylim([0, 1])

def get_spikes_figure_and_subplots(n_pops):
    height_ratios = [4]
    for i in range(n_pops):
        height_ratios += [4, 1]
    return plt.subplots(
        (n_pops*2) + 1, sharex=True, layout="constrained", gridspec_kw={'height_ratios': height_ratios})


def plot_spikes(pops, t_start, t_end, data_manager, show = False):
    ra_times, ra = data_manager.get_data_window(("or", "ra"), t_start, t_end)
    most_active_or = data_manager.or_most_active(ra)

    figure, subplots = get_spikes_figure_and_subplots(len(pops))

    plot_ra(most_active_or, ra_times, ra, subplots[0])

    for (i, pop) in enumerate(pops):
        time, voltage = data_manager.get_data_for_first_neuron_in_glomerulus(
                most_active_or,
                pop,
                'V',
                t_start,
                t_end
                )

        spike_times = data_manager.get_spikes_for_first_neuron_in_glomerulus(
                most_active_or,
                pop,
                t_start,
                t_end
                )

        neuron_idx = data_manager.get_first_neuron_in_glomerulus(
                most_active_or,
                pop
                )

        plot_voltage(
            voltage=voltage,
            time=time,
            spike_times=spike_times,
            pop_name=pop,
            id_neuron=neuron_idx,
            subplot=subplots[i*2+1],
            kernel_dimension=10
        )

        plot_spikes_for_population(spike_times, subplots[i*2+2])


    filename = f"spikes/{t_start:.1f}_{t_end:.1f}.png"

    data_manager.show_or_save(filename, show)

if __name__ == "__main__":
    from beegenn.parameters.reading_parameters import parse_cli
    from pathlib import Path
    import pandas as pd
    param = parse_cli()
    data_manager = DataManager(param['simulations']['simulation'], param['simulations']
                 ['name'], param['neuron_populations'], param['synapses'])

    events = pd.read_csv(Path(param['simulations']['simulation']['output_path']) / param['simulations']['name'] / 'events.csv')

    plot_spikes(['orn'], 0, np.min(events['t_end']), data_manager, show = False)
