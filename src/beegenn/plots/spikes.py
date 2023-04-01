import numpy as np

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

def plot_spikes(spike_times, subplot):
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
