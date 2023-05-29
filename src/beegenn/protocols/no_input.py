from .protocol import Protocol
import numpy as np


class NoInput(Protocol):
    """
    Creates the events for the first experiment.
    Namely each odor is presented at a different number of concentration steps
    in a sequential manner.
    Each odor presentation lasts 3s and they are separated by a 3s resting interval.

    ...


    Methods
    -------
    events_generation(num_concentration_increases)
        Takes as input the number of times the concentration for each odor is increased
        and creates the corresponding events.
    """

    def __init__(self, param):
        super().__init__(param)
        self.max_time = param['simulation_time']



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
        concentration = self.starting_concentration*np.power(self.dilution_factor, c_exp) #Concentration is increased by increasing it by a dilution factor
        event = {
            "t_start" : t_start,
            "t_end" : t_end,
            "concentration" : concentration,
            "odor_name" : odor.name,
            "binding_rates" : np.power(odor.binding_rates*concentration, self.hill_exponential), #Binding rates are updated so to include inforamtion about concentration
            "activation_rates" : odor.activation_rates ,
            "happened" : False,
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
        for (i, odor) in enumerate(self.odors):
            for c_exp in range(num_concentration_increases):
                res.append(self._event_generation(t, odor, c_exp))
                t = res[-1]['t_end'] + self.resting_duration
        self.events = res

if __name__ == "__main__":
    from reading_parameters import get_parameters
    import sys
    params = get_parameters(sys.argv[1])
    temp = FirstProtocol(params['protocols']['experiment1'], 1)
    temp.generate_or_param(params['neuron_populations']['or'])
    temp.generate_inhibitory_connectivity(25 * 160, 5 * 160, sys.argv[2], False)
