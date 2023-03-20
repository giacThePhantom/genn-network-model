from protocol import Protocol
import numpy as np

class ThirdProtocol(Protocol):
    def __init__(self, param):
        super().__init__(param)
        self.events_generation()
        self.assign_channel_to_events()

    def _event_generation(self, t, odor, c_exp):
        t_start = t
        t_end = t + self.event_duration
        concentration = c_exp
        event = {
            "t_start" : t_start,
            "t_end" : t_end,
            "concentration" : concentration,
            "odor_name" : odor.name,
            "binding_rates" : np.power(odor.binding_rates*concentration, self.hill_exponential),
            "activation_rates" : odor.activation_rates,
            "happened" : False,
        }
        return event

    def events_generation(self):
        res = []
        t = self.resting_duration
        for c1_exp in self.param['first_odor_concentrations']:
            for c2_exp in self.param['second_odor_concentrations']:
                # apply IAA and Geo at the same time, but with different concentrations
                res.append(self._event_generation(t, self.odors[1], c1_exp))
                res.append(self._event_generation(t, self.odors[2], c2_exp))
                t = res[-1]['t_end'] + self.resting_duration

        self.events = res
        self.assign_channel_to_events()
