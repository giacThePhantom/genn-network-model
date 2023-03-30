import tables
from pathlib import Path
import pickle
import numpy as np
from . import spikes
import matplotlib.pyplot as plt
from beegenn.reading_parameters import parse_cli


class Plots:

    def __init__(self, sim_param, sim_name, neuron_param, synapse_param):
        self.sim_param = sim_param
        self.neuron_param = neuron_param
        self.synapse_param = synapse_param
        self._root_out_dir = Path(sim_param['output_path']) / sim_name
        self._root_plot_dir = self._root_out_dir / 'plots'
        self._root_plot_dir.mkdir(exist_ok = True)
        self.data = tables.open_file(str(self._root_out_dir / 'tracked_vars.h5'))
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
            timestep_start = np.searchsorted(data[:,0], t_start)
            timestep_end = np.searchsorted(data[:,0], t_end, 'right')
        else:
            timestep_start = np.floor(t_start/(self.sim_param['dt']*self.sim_param['n_timesteps_to_pull_var']))
            timestep_end = np.ceil(t_end/(self.sim_param['dt']*self.sim_param['n_timesteps_to_pull_var']))
        data = data[timestep_start:timestep_end]
        return data[:,0], np.squeeze(data[:,1:])


    def pick_time_window(self, event_index, include_resting = False, only_resting = False):
        event = self.protocol.events[event_index]
        t_start = event['t_start']
        t_end = event['t_end']
        if only_resting:
            t_start = t_end
        if include_resting:
            t_end += self.protocol.resting_duration

        return t_start, t_end

    def plot_spikes(self, pops, event_index, include_resting = False, only_resting = False, show = False):

        t_start, t_end = self.pick_time_window(event_index, include_resting, only_resting)
        ra_times, ra = self.get_data_window(("or", "ra"), t_start, t_end)
        most_active_or = spikes.or_most_active(ra)

        figure, subplots = plt.subplots(len(pops) + 1, sharex=True, layout="constrained")

        spikes.plot_ra(most_active_or, ra_times, ra, subplots[0])

        for (pop, subplot) in zip(pops, subplots[1:]):
            time, voltage = self.get_data_window((pop, "V"), t_start, t_end)
            neuron_idx = most_active_or * self.neuron_param[pop]['n'] // self.neuron_param['or']['n']
            voltage = voltage[:, neuron_idx]

            spike_times, spike_id = self.get_data_window((pop, "spikes"), t_start, t_end)
            filtered_spike_idx = spike_id == neuron_idx
            spike_times = spike_times[filtered_spike_idx]

            spikes.plot_spikes(
                    voltage = voltage,
                    time = time,
                    spike_times = spike_times,
                    pop_name = pop,
                    id_neuron = neuron_idx,
                    subplot = subplot,
                    kernel_dimension = 10
                    )

        filename = self._root_plot_dir / 'spikes' / f"{t_start:.1f}_{t_end:.1f}.png"
        filename.parent.mkdir(exist_ok = True)
        self._show_or_save(filename, show)

        pass


    def _show_or_save(self, filename, show = False):
        if show:
            plt.show()
        else:
            plt.savefig(filename, dpi = 700, bbox_inches = 'tight')
        plt.cla()
        plt.clf()


if __name__ == '__main__':
    param = parse_cli()
    temp = Plots(param['simulations']['simulation'], param['simulations']['name'], param['neuron_populations'], param['synapses'])

    temp.plot_spikes(['orn', 'pn', 'ln'], 2, show = True, include_resting = True)
