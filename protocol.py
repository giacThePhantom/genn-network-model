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


def event_generation(protocol, t, c_exp, odor):
    t_start = t
    t_end = t + protocol.egt_event_duration()

def events_generation(protocol: Protocol, num_concentration_increases):
    res = []
    t = protocol.get_resting_duration()
    for (i, odor) in enumerate(protocol.get_odors()):
        for c_exp in range(num_concentration_increases):
            t_start = t
            t_end = t + protocol.get_event_duration()
            t = t_end + protocol.get_resting_duration()
            concentration = protocol.get_starting_concentration()*np.power(protocol.get_dilution_factor(), c_exp)
            event = {
                "t_start" : t_start,
                "t_end" : t_end,
                "concentration" : concentration,
                "odor_name" : odor.get_name(),
                "binding_rates" : np.power(odor.get_binding_rates()*concentration, protocol.get_hill_exponential()),
                "activation_rates" : odor.get_activation_rates() ,
            }
            res.append(event)
    return res


if __name__ == "__main__":
    from reading_parameters import get_parameters
    import sys
    params = get_parameters(sys.argv[1])
    temp = Protocol(params['protocols']['experiment1'])
    events = event_generation(temp, 2)
