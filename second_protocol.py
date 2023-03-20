from protocol import Protocol
import numpy as np

class SecondProtocol(Protocol):
    """
    Creates the events for the second experiment.
    Namely two odors are presented at the same time, with a varying concentration
    in a sequential manner.

    ...


    Methods
    -------
    events_generation(num_concentration_increases)
        Takes as input the number of times the concentration for each odor is increased
        and creates the corresponding events.
    """

    def __init__(self, param):
        super().__init__(param)
        self.events_generation(param['concentration_increases'])
        self.assign_channel_to_events() #Assign an odor to a channel in the OR population

    def _event_generation(self, t, odor, c_exp):
        """Builds an event
        Parameters
        ----------
        t : double
            The time at which an event starts
        odor : Odor
            The object containing the activation and binding rates of the particular odor
        c_exp : int
            The number of time the concentration is increased
        Return
        ------
        event : dict
            A dictionary containing all the information necessary for event
            characterization
        """

        t_start = t
        t_end = t + self.event_duration
        concentration = self.starting_concentration*np.power(self.dilution_factor, c_exp)
        event = {
            "t_start" : t_start,
            "t_end" : t_end,
            "concentration" : concentration,
            "odor_name" : odor.name,
            "binding_rates" : np.power(odor.binding_rates*concentration, self.hill_exponential),
            "activation_rates" : odor.activation_rates ,
            "happened": False
        }
        return event

    def events_generation(self, num_concentration_increases):
        """Creates the event for the protocol and saves them in a private field
        Parameters
        ----------
        num_concentration_increases : int
            The number of times the concentration is increased by a dilution factor
        """

        res = []
        t = self.resting_duration
        #for (i, odor) in enumerate(self.get_odors()):
        odors = self.odors
        for c1_exp in range(num_concentration_increases):
            for c2_exp in range(num_concentration_increases):
                # apply two odors at the same time, but with different concentrations.
                res.append(self._event_generation(t, odors[1], c1_exp))
                res.append(self._event_generation(t, odors[2], c2_exp))
                t = res[-1]['t_end'] + self.resting_duration

        self.events = res
        self.assign_channel_to_events()

if __name__ == "__main__":
    from reading_parameters import get_parameters
    import sys
    params = get_parameters(sys.argv[1])
    temp = SecondProtocol(params['protocols']['experiment2'])
    print(len(temp.get_events()))
    print(temp.get_simulation_time())
