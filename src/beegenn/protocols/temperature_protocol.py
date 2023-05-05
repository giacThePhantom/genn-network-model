import numpy as np

class TemperatureProtocol:

    def __init__(self, param):
        self.param = param
        self.max_time = None
        self.connectivity_matrix = self._generate_inhibitory_connectivity(
                param['connectivity_type'],
                param['self_inhibition']
                )
        self.events = self.events_generation()
        self.events.sort(key = lambda x : x['t_start'])
        self.simulation_time = self.events[-1]['t_end']

    def _generate_inhibitory_connectivity(self, connectivity_type, self_inhibition):
        if connectivity_type == 'homogeneous':
            connectivity_matrix = np.ones(
                    (
                        self.param['num_glomeruli'],
                        self.param['num_glomeruli']
                        )
                    )
        else:
            connectivity_matrix = np.zeros(
                    (
                        self.param['num_glomeruli'],
                        self.param['num_glomeruli']
                        )
                    )
        if not self_inhibition:
            np.fill_diagonal(connectivity_matrix, 0)
        return connectivity_matrix

    def add_inhibitory_conductance(self, param, n_source, n_target):
        """Adds the connectivity matrix to a population of synapses
        Parameters
        ----------
        param : dict
            The parameter which will later be used to build the synapses' population.
        n_source : int
            The number of pre-synaptic neurons
        n_target : int
            The number of post-synaptic neurons
        """

        connectivity_matrix = self.connectivity_matrix * param['wu_var_space']['g']
        n_glomeruli = connectivity_matrix.shape[1]
        connectivity_matrix = np.repeat(connectivity_matrix, repeats = n_source / n_glomeruli, axis = 0)
        connectivity_matrix = np.repeat(connectivity_matrix, repeats = n_target / n_glomeruli, axis = 1)
        param['wu_var_space']['g'] = connectivity_matrix.flatten()

    def _event_generation(self, time, temperature):
        event = {
                "t_start" : time,
                "t_end" : time + self.param['event_duration'] + self.param['resting_duration'],
                "name" : "T=" + str(temperature),
                "affects" : self.param['affects'],
                "happened" : False
                }
        values = {}
        for pop in event["affects"]:
            values[pop] = {}
            for var in event["affects"][pop]:
                values[pop][var] = temperature
        event['affects'] = values
        return event

    def events_generation(self):
        """Generates the events for the temperature
        Returns
        -------
        events : dict
            A dictionary containing the events for the inhibitory synapses
        """
        events = []
        temperature = self.param['starting_temperature']
        time = 0.0
        while temperature <= self.param['ending_temperature']:
            events.append(
                    self._event_generation(
                        time,
                        temperature
                        )
                    )
            temperature += self.param['temperature_step']
            time = time + self.param['event_duration'] + self.param['resting_duration']
        return events
