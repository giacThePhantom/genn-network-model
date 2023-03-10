from protocol import Protocol
import numpy as np


class FirstProtocol(Protocol):

    def _event_generation(self, t, odor, c_exp):
        t_start = t
        t_end = t + self.get_event_duration()
        concentration = self.get_starting_concentration()*np.power(self.get_dilution_factor(), c_exp)
        event = {
            "t_start" : t_start,
            "t_end" : t_end,
            "concentration" : concentration,
            "odor_name" : odor.get_name(),
            "binding_rates" : np.power(odor.get_binding_rates()*concentration, self.get_hill_exponential()),
            "activation_rates" : odor.get_activation_rates() ,
        }
        return event

    def events_generation(self, num_concentration_increases):
        res = []
        t = self.get_resting_duration()
        for (i, odor) in enumerate(self.get_odors()):
            for c_exp in range(num_concentration_increases):
                res.append(self._event_generation(t, odor, c_exp))
                t = res[-1]['t_end'] + self.get_resting_duration()
        self.events = res
        self.assign_channel_to_events()

if __name__ == "__main__":
    from reading_parameters import get_parameters
    import sys
    params = get_parameters(sys.argv[1])
    temp = FirstProtocol(params['protocols']['experiment1'])
    temp.events_generation(1)
    temp.assign_channel_to_events()
    temp.generate_or_param(params['neuron_populations']['or'])
