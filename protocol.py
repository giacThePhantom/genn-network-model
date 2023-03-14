from odors import Odor
import numpy as np
from abc import ABC, abstractmethod
import pygenn.genn_wrapper.Models as var_access

class Protocol(ABC):

    def _create_odor(self, param, name):
        return Odor(param, name, self.param['num_glomeruli'], self.param['homogeneous'])

    def _create_odors(self, odor_params):
        res = []
        for i in self.param['odors']:
            res.append(self._create_odor(odor_params[i], i))
        while len(res) < self.param['num_odors']:
            res.append(self._create_odor(odor_params['default'], 'default'))
        return res

    def _odor_binding_rate_permutation(self):
        not_default_shuffle = np.arange(self.param['num_glomeruli'])
        for i in self.odors:
            if i.get_name() != 'default':
                i.shuffle_binding_rates(not_default_shuffle)
            else:
                i.shuffle_binding_rates()

    def _compute_hill_exponential(self, param):
        if isinstance(param, dict):
            res = np.random.uniform(param['min'], param['max'], self.param['num_glomeruli'])
        else:
            res = param
        return res

    def set_events(self, events):
        self.events = events

    def get_odors(self):
        return self.odors

    def get_starting_concentration(self):
        return self.param['concentration']['start']

    def get_dilution_factor(self):
        return np.power(self.param['concentration']['dilution_factor']['base'], self.param['concentration']['dilution_factor']['exponent'])

    def get_event_duration(self):
        return self.param['event_duration']

    def get_resting_duration(self):
        return self.param['resting_duration']

    def get_hill_exponential(self):
        return self.param['hill_exponential']

    def get_events(self):
        return self.events

    def assign_channel_to_events(self):
        channel_occupancy_state = [[] for i in range(self.param['num_channels'])]
        for i in self.events:
            found_free_channel = False
            channel_index = 0
            while not found_free_channel and channel_index < self.param['num_channels']:
                channel_occupied_during_event = False
                for j in channel_occupancy_state[channel_index]:
                    if (i['t_start'] >= j[0] and i['t_start'] <= j[1]) or (i['t_end'] >= j[0] and i['t_end'] <= j[1]):
                        channel_occupied_during_event = True
                if not channel_occupied_during_event:
                    found_free_channel = True
                else:
                    channel_index += 1
            if found_free_channel:
                i['channel'] = channel_index
                channel_occupancy_state[channel_index].append((i['t_start'], i['t_end']))
            else:
                raise Exception("The number of channels is not enough to allow for all the events to happen")

    def _generate_inhibitory_connectivity(self, connectivity_type, self_inhibition):
        binding_rates_matrix = np.array([i.get_binding_rates() for i in self.odors])
        if connectivity_type == 'correlation':
            connectivity_matrix = np.corrcoef(binding_rates_matrix,rowvar=False)
            connectivity_matrix = (connectivity_matrix + 1.0)/20.0
            pass
        elif connectivity_type == 'covariance':
            connectivity_matrix= np.cov(binding_rates_matrix,rowvar=False)
            connectivity_matrix= np.maximum(0.0, connectivity_matrix)
            pass
        elif connectivity_type == 'homogeneous':
            connectivity_matrix = np.ones((binding_rates_matrix.shape[1],
                                            binding_rates_matrix.shape[1]))
            pass

        if not self_inhibition:
            np.fill_diagonal(connectivity_matrix, 0)
        return connectivity_matrix

    def add_inhibitory_conductance(self, param, n_source, n_target):
        connectivity_matrix = self.connectivity_matrix * param['wu_var_space']['g']
        n_glomeruli = connectivity_matrix.shape[1]
        connectivity_matrix = np.repeat(connectivity_matrix, repeats = n_source / n_glomeruli, axis = 0)
        connectivity_matrix = np.repeat(connectivity_matrix, repeats = n_target / n_glomeruli, axis = 1)
        param['wu_var_space']['g'] = connectivity_matrix.flatten()

    def get_simulation_time(self):
        max_time = max(self.events, key = lambda x : x['t_end'])['t_end']
        return max_time + self.param['resting_duration']


    @abstractmethod
    def _event_generation(self):
        pass

    @abstractmethod
    def events_generation(self):
        pass

    def get_events_for_channel(self):
        res = [[] for i in range(self.param['num_channels'])]

        for i in self.events:
            res[i['channel']].append(i)

        for i in res:
            i.sort(key = lambda x : x['t_start'])

        return res


    def __init__(self, param):
        self.param = param
        self.odors = self._create_odors(self.param['odors'])
        self._odor_binding_rate_permutation()
        self.param['hill_exponential'] = self._compute_hill_exponential(self.param['hill_exponential'])
        self.events = []
        self.connectivity_matrix = self._generate_inhibitory_connectivity(param['connectivity_type'], param['self_inhibition'])

if __name__ == "__main__":
    from reading_parameters import get_parameters
    import sys
    params = get_parameters(sys.argv[1])
    temp = Protocol(params['protocols']['experiment1'])
    events = event_generation(temp, 2)
