from protocol import Protocol
import numpy as np

def exp2_protocol(n_conc=25, odor1=0, odor2=1):
    """
    Apply both the first and second odor at the same time for 3 seconds, but with
    different concentrations. Exaustively try all N^2 pairs. Make a 3-second break
    after exposure.
    """
    protocol = []
    base = np.power(10, 1/4) # every 4 steps we have a decuplication
    t_start = 0.0
    for c1 in range(n_conc):
        for c2 in range(n_conc):
            protocol.append(
                ProtocolStep(
                    t_start,
                    t_start + 3000.0,
                    odor1,
                    1e-7*np.power(base, c1),
                )
            )
            protocol.append(
                ProtocolStep(
                    t_start,
                    t_start + 3000.0,
                    odor2,
                    1e-7*np.power(base, c2),
                )
            )
            protocol.append(
                ProtocolStep(
                    t_start + 3000.0,
                    t_start + 6000.0,
                    odor1,
                    0.0
                )
            )
            protocol.append(
                ProtocolStep(
                    t_start + 3000.0,
                    t_start + 6000.0,
                    odor2,
                    0.0
                )
            )
            t_start += 6000
    return protocol

class SecondProtocol(Protocol):
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

    def events_generation(self, num_concentration_increases=25):
        res = []
        t = self.get_resting_duration()
        #for (i, odor) in enumerate(self.get_odors()):
        odors = self.get_odors()
        for c1_exp in range(num_concentration_increases):
            for c2_exp in range(num_concentration_increases):
                # apply two odors at the same time, but with different concentrations.
                res.append(self._event_generation(t, odors[0], c1_exp))
                res.append(self._event_generation(t, odors[1], c2_exp))
                t = res[-1]['t_end'] + self.get_resting_duration()

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
