import tables
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from beegenn.reading_parameters import parse_cli
import pandas as pd

from . import spikes
from . import sdf
from . import mean_activation_glomerulus as mag


class Plots:

    def __init__(self, sim_param, sim_name, neuron_param, synapse_param):
        self.sim_param = sim_param
        self.neuron_param = neuron_param
        self.synapse_param = synapse_param
        self._root_out_dir = Path(sim_param['output_path']) / sim_name
        self._root_plot_dir = self._root_out_dir / 'plots'
        self._root_plot_dir.mkdir(exist_ok=True)
        self.data = tables.open_file(
            str(self._root_out_dir / 'tracked_vars.h5'))
        self.recorded_data = sim_param['tracked_variables']
        with (self._root_out_dir / 'protocol.pickle').open('rb') as f:
            self.protocol = pickle.load(f)
        pass

    def _parse_data(self):
        pass

    def _load_data(self):
        pass

    def get_data_window(self, var_path, t_start, t_end):
        path = '/' + '/'.join(var_path)
        data = self.data.root[path]
        if var_path[-1] == 'spikes':
            # horrible hack to read the entire EArray as a numpy array.
            copied_data = np.squeeze(data[:])
            timesteps = data[:, 0]
            timesteps_start = timesteps >= t_start
            timesteps_end = timesteps < t_end
            filtered = np.where(timesteps_start & timesteps_end)
            filtered_timesteps = np.squeeze(copied_data[filtered, 0])
            filtered_data = np.squeeze(copied_data[filtered, 1:])

            return filtered_timesteps, filtered_data

        else:
            timestep_start = np.floor(
                t_start/(self.sim_param['dt']*self.sim_param['n_timesteps_to_pull_var']))
            timestep_end = np.ceil(
                t_end/(self.sim_param['dt']*self.sim_param['n_timesteps_to_pull_var']))
            data = data[timestep_start:timestep_end]
            return data[:, 0], np.squeeze(data[:, 1:])

    def pick_time_window(self, event_index, include_resting=False, only_resting=False):
        event = self.protocol.events[event_index]
        t_start = event['t_start']
        t_end = event['t_end']
        if only_resting:
            t_start = t_end
        if include_resting:
            t_end += self.protocol.resting_duration

        return t_start, t_end

    def plot_spikes(self, pops, t_start, t_end, show=False):
        ra_times, ra = self.get_data_window(("or", "ra"), t_start, t_end)
        most_active_or = spikes.or_most_active(ra)

        height_ratios = [4]
        for i in pops:
            height_ratios += [4, 1]
        figure, subplots = plt.subplots(
            (len(pops)*2) + 1, sharex=True, layout="constrained", gridspec_kw={'height_ratios': height_ratios})

        spikes.plot_ra(most_active_or, ra_times, ra, subplots[0])

        for (i, pop) in enumerate(pops):
            time, voltage = self.get_data_window((pop, "V"), t_start, t_end)
            neuron_idx = most_active_or * \
                self.neuron_param[pop]['n'] // self.neuron_param['or']['n']
            voltage = voltage[:, neuron_idx]

            spike_times, spike_id = self.get_data_window(
                (pop, "spikes"), t_start, t_end)
            filtered_spike_idx = spike_id == neuron_idx
            spike_times = spike_times[filtered_spike_idx]

            spikes.plot_voltage(
                voltage=voltage,
                time=time,
                spike_times=spike_times,
                pop_name=pop,
                id_neuron=neuron_idx,
                subplot=subplots[i*2+1],
                kernel_dimension=10
            )

            spikes.plot_spikes(spike_times, subplots[i*2+2])

        filename = self._root_plot_dir / 'spikes' / \
            f"{t_start:.1f}_{t_end:.1f}.png"
        filename.parent.mkdir(exist_ok=True)
        self._show_or_save(filename, show)

    def get_spike_matrix(self, spike_times, spike_ids, pop, t_start, t_end):
        duration_timesteps = int(
            np.ceil((t_end-t_start)/self.sim_param['dt'])) + 1
        res = np.zeros((self.neuron_param[pop]['n'], duration_timesteps))

        for (i, (time, id)) in enumerate(zip(spike_times, spike_ids)):
            time = int((time - t_start)/self.sim_param['dt'])
            res[int(id)][time] = 1.0

        return res

    def plot_sdf_heatmap(self, pops, t_start, t_end, show):

        figure, subplots = plt.subplots(
            1, len(pops), sharey=True, layout="constrained")
        image = []

        for (pop, subplot) in zip(pops, subplots):
            spike_times, spike_ids = self.get_data_window(
                (pop, "spikes"), t_start, t_end)
            spike_matrix = self.get_spike_matrix(
                spike_times, spike_ids, pop, t_start, t_end)
            sdf_matrix = sdf.compute_sdf_for_population(
                spike_matrix, self.sim_param['sdf_sigma'], self.sim_param['dt'])
            sdf_matrix_avg = sdf.sdf_glomerulus_avg(
                sdf_matrix, self.neuron_param['or']['n'])
            image.append(sdf.plot_sdf_heatmap(sdf_matrix_avg,
                         t_start, t_end, self.sim_param['dt'], pop, subplot))

        cbar = figure.colorbar(image[-1], ax=subplots[-1])
        cbar.ax.set_ylabel("SDF ($Hz$)")

        filename = self._root_plot_dir / 'sdf' / \
            f"{t_start:.1f}_{t_end:.1f}.png"
        filename.parent.mkdir(exist_ok=True)

        self._show_or_save(filename, show)

    def plot_outliers_sdf_over_time(self, pops, t_start, t_end, show):
        figure, subplots = plt.subplots(len(pops))

        for (pop, subplot) in zip(pops, subplots):
            spike_times, spike_ids = self.get_data_window(
                (pop, "spikes"), t_start, t_end)
            spike_matrix = self.get_spike_matrix(
                spike_times, spike_ids, pop, t_start, t_end)
            sdf_matrix = sdf.compute_sdf_for_population(
                spike_matrix, self.sim_param['sdf_sigma'], self.sim_param['dt'])
            sdf_matrix_avg = sdf.sdf_glomerulus_avg(
                sdf_matrix, self.neuron_param['or']['n'])

            mag.plot_active_glomeruli_sdf(sdf_matrix_avg, subplot)

        filename = self._root_plot_dir / 'sdf_over_time' / \
            f"{t_start:.1f}_{t_end:.1f}.png"
        filename.parent.mkdir(exist_ok=True)
        self._show_or_save(filename, show)

    def plot_sdf_over_c(self, pops, t_start, t_end, show):
        pass

    def _show_or_save(self, filename, show=False):
        print("Saving to ", filename)
        if show:
            plt.show()
        else:
            plt.savefig(filename, dpi=700, bbox_inches='tight')
        plt.cla()
        plt.clf()
        plt.close()

    def close_file(self):
        self.data.close()


if __name__ == '__main__':
    param = parse_cli()
    temp = Plots(param['simulations']['simulation'], param['simulations']
                 ['name'], param['neuron_populations'], param['synapses'])

    events = pd.read_csv(Path(param['simulations']['simulation']['output_path']) / param['simulations']['name'] / 'events.csv')
    events = list(dict.fromkeys((start, end) for (start, end) in zip(events['t_start'], events['t_end'])))
    # events = [(12000*i + 3000, 12000.0*i + 6000.0) for i in range(12)]

    print(events)

    for (t_start, t_end) in events:
        temp.plot_sdf_heatmap(['orn', 'pn', 'ln'], t_start, t_end, show=False)
        temp.plot_outliers_sdf_over_time(['orn', 'pn', 'ln'], t_start, t_end, show = False)

    # temp.plot_spikes(['orn', 'pn', 'ln'], 3000, 9000, show = False)
    # temp.plot_sdf_heatmap(['orn', 'pn', 'ln'], 3000, 9000, show = False)
    # temp.plot_sdf_heatmap(['orn', 'pn', 'ln'], 3000, 6000, show = False)
    # temp.plot_mean_activation_glomerulus(['orn', 'pn', 'ln'], 0, 12000, show = False)
    temp.close_file()
