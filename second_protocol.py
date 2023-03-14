from protocol import Protocol
import numpy as np

class SecondProtocol(Protocol):
    def _event_generation(self, t, odor, c_exp):
        t_start = t
        t_end = t + self.event_duration
        concentration = self.starting_concentration*np.power(self.dilution_factor, c_exp)
        event = {
            "t_start" : t_start,
            "t_end" : t_end,
            "concentration" : concentration,
            "odor_name" : odor.get_name(),
            "binding_rates" : np.power(odor.get_binding_rates()*concentration, self.hill_exponential),
            "activation_rates" : odor.get_activation_rates() ,
        }
        return event

    def events_generation(self, num_concentration_increases=25):
        res = []
        t = self.resting_duration
        #for (i, odor) in enumerate(self.get_odors()):
        odors = self.odors
        for c1_exp in range(num_concentration_increases):
            for c2_exp in range(num_concentration_increases):
                # apply two odors at the same time, but with different concentrations.
                res.append(self._event_generation(t, odors[0], c1_exp))
                res.append(self._event_generation(t, odors[1], c2_exp))
                t = res[-1]['t_end'] + self.resting_duration

        self.events = res
        self.assign_channel_to_events()

if __name__ == "__main__":
    from reading_parameters import get_parameters
    import sys
    params = get_parameters(sys.argv[1])
    temp = SecondProtocol(params['protocols']['experiment2'])
    temp.events_generation(1)
    temp.assign_channel_to_events()
    temp.generate_or_param(params['neuron_populations']['or'])
