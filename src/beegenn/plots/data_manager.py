import tables
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from beegenn.parameters.reading_parameters import parse_cli
import pandas as pd
import scipy as sp

class DataManager:

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
        self.events = pd.read_csv(self._root_out_dir / 'events.csv')
        with (self._root_out_dir / 'protocol.pickle').open('rb') as f:
            self.protocol = pickle.load(f)

    def close(self):
        self.data.close()

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

    def get_data_for_first_neuron_in_glomerulus(self, glo_idx, pop, var, t_start, t_end):
        neuron_idx = self.get_first_neuron_in_glomerulus(glo_idx, pop)
        time, voltage = self.get_data_window((pop, var), t_start, t_end)
        voltage = voltage[:, neuron_idx]
        return time, voltage

    def get_spikes_for_first_neuron_in_glomerulus(self, glo_idx, pop, t_start, t_end):
        neuron_idx = self.get_first_neuron_in_glomerulus(glo_idx, pop)
        spike_times, spike_id = self.get_data_window(
            (pop, "spikes"), t_start, t_end)
        filtered_spike_idx = spike_id == neuron_idx
        spike_times = spike_times[filtered_spike_idx]
        return spike_times

    def get_first_neuron_in_glomerulus(self, glo_idx, pop):
        neuron_idx = glo_idx * \
            self.neuron_param[pop]['n'] // self.neuron_param['or']['n']
        return neuron_idx

    def or_most_active(self, ra):
        #Sum over time
        ra_sum = np.sum(ra, axis=0)
        #Pick the or that has been the most active during
        #The time window
        or_most_active = np.argmax(ra_sum)
        return or_most_active


    def get_spike_matrix(self, spike_times, spike_ids, pop, t_start, t_end):
        duration_timesteps = int(
            np.ceil((t_end-t_start)/self.sim_param['dt'])) + 1
        res = np.zeros((self.neuron_param[pop]['n'], duration_timesteps))

        for (time, id) in zip(spike_times, spike_ids):
            time = int((time - t_start)/self.sim_param['dt'])
            res[int(id)][time] = 1.0

        return res

    def sdf_for_population(self, pop, t_start, t_end):
        spike_times, spike_ids = self.get_data_window(
                (pop, 'spikes'),
                t_start,
                t_end,
                )
        spike_matrix = self.get_spike_matrix(
                spike_times,
                spike_ids,
                pop,
                t_start,
                t_end
                )
        sigma = self.sim_param['sdf_sigma']
        dt = self.sim_param['dt']
        kernel = np.arange(-3*sigma, +3*sigma, dt)
        kernel = np.exp(-np.power(kernel, 2)/(2*sigma*sigma))
        kernel = kernel/(sigma*np.sqrt(2.0*np.pi))*1000.0
        sdf = np.apply_along_axis(
                lambda m : sp.signal.convolve(m, kernel, mode='same'),
                axis = 1,
                arr=spike_matrix)
        return sdf

    def sdf_per_glomerulus_avg(self, pop, t_start, t_end):
        """
        Get the average activation intensity across n-sized groups of glomeruli
        for each timestep.
        """
        sdf = self.sdf_for_population(pop, t_start, t_end)
        n_glomeruli = self.neuron_param['or']['n']
        res = np.zeros((n_glomeruli, sdf.shape[1]))
        glomerulus_dim = sdf.shape[0] // n_glomeruli
        for i in range(n_glomeruli):
            res[i, :]= np.mean(sdf[glomerulus_dim*i:glomerulus_dim*(i+1), :],axis=0)
        return res

    def get_sim_dt(self):
        return self.sim_param['dt']


    def show_or_save(self, filename, show=False):
        filename = self._root_plot_dir / filename
        filename.parent.mkdir(exist_ok=True)
        if show:
            plt.show()
        else:
            print("Saving to ", filename)
            plt.savefig(filename, dpi=700, bbox_inches='tight')
        plt.cla()
        plt.clf()
        plt.close()

    def get_pop_size(self, pop):
        return self.neuron_param[pop]['n']

    def get_inhibitory_connectivity(self):
        connectivity_matrix = self.protocol._generate_inhibitory_connectivity(self.protocol.param['connectivity_type'], self.protocol.param['self_inhibition'])
        return connectivity_matrix

    def get_events(self):
        return self.events
