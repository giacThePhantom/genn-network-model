from odors import Odor
import numpy as np
from abc import ABC, abstractmethod

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
                raise Exception("The number of channel is not enough to allow for all the events to happen")


    def generate_or_param(self, or_params):
        per_slot_events = [[] for i in range(self.param['num_channels'])]

        for i in self.events:
            per_slot_events[i['channel']].append(i)

        sim_code_to_be_added = ""
        for i in per_slot_events:
            for (j, event) in enumerate(i):
                var_binding_rate = "kp1cn_" + str(event['channel'])
                new_binding_var = {
                    "name" : var_binding_rate + "_" + str(j),
                    "type" : "scalar",
                    "value" : list(event['binding_rates'])
                }
                or_params['variables'].append(new_binding_var)
                var_activation_rate = "kp2_" + str(event['channel'])
                new_activation_var = {
                    "name" : var_activation_rate + '_' + str(j),
                    "type" : "scalar",
                    "value" : list(event['activation_rates'])
                }
                or_params['variables'].append(new_activation_var)
                sim_code_to_be_added += f"if ($(t) >= {event['t_start']} && $(t) <= {event['t_end']}) {{$({var_binding_rate}) = $({new_binding_var['name']}); $({var_activation_rate}) = $({new_activation_var['name']}); }} else "

            if len(i) > 0:
                sim_code_to_be_added += f"{{$({var_binding_rate}) = 0;}}"

        if isinstance(or_params['sim_code'], str):
            or_params['sim_code'] = sim_code_to_be_added + '\n' + or_params['sim_code']
        else:
            or_params['sim_code'].insert(0, sim_code_to_be_added)

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
        return connectivity_matrix.flatten()

    def generate_inhibitory_conductance(self, ln_pn_param):
        connectivity_matrix = self.connectivity_matrix * ln_pm_param['wu_var_space']['g']
        ln_pm_param['wu_var_space']['g'] = connectivity_matrix
        connectivity_matrix = np.repeat(connectivity_matrix, repeats = n_source / binding_rates_matrix.shape[1], axis = 0)
        connectivity_matrix = np.repeat(connectivity_matrix, repeats = n_target / binding_rates_matrix.shape[1], axis = 1)

    @abstractmethod
    def _event_generation(self):
        pass

    @abstractmethod
    def events_generation(self):
        pass


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
