from .odors import Odor

import numpy as np
from abc import ABC, abstractmethod

class Protocol(ABC):
    """
    A class for a protocol, it builds the events that happen during the
    simulation

    ...

    Attributes
    ----------
    param : dict
        A dictionary containing all the protocol's parameters
    events : list
        A list of all of the events of the protocol
    connectivity_matrix : np.ndArray
        A matrix containing the inhibitory connectivity of the network for
        which this protocol is built
    Methods
    -------
    odors() : list
        Returns the list of odors built during the protocol
    starting_concentration() : double
        Returns the starting concentration of odors
    dilution_factor() : double
        Returns the dilution factor of the concentration
    event_duration() : double
        Returns the duration of a single event
    resting_duration() : double
        Returns how much time passes between events
    hill_exponential() : double
        Returns the hill exponential
    assign_channel_to_events() : void
        Assigns events to a channel so that a channel is presented at most one
        odor at a time
    add_inhibitory_conductance(param, n_source, n_target) : np.ndArray
        Adds to the parameters of a synapses the inhibitory conductances
    simulation_time() : double
        Returns the simulation time
    events_generation() : void
        Abstract method that generates the events
    get_events_for_channel() : list
        Returns all the events split into lists such that each list
        contains all events that involve a single channel
    get_events() : list
        Returns all the events
    """

    def _create_odor(self, param, name):
        """Creates an odor
        Parameters
        ----------
        param : dict
            A dictionary for the odor parameters
        name : str
            The name of the odor
        Returns
        -------
        res : Odor
            The corresponding odor, containing binding and activation rates
        """

        return Odor(param, name, self.param['num_glomeruli'], self.param['homogeneous'])

    def _create_odors(self, odor_params):
        """Creates num_odors odors for the protocol, if the odors are less than
        what is required, the list is padded with the odor named 'default'
        Parameters
        ----------
        odor_params : dict
            A dictionary containing all the parameters for each odor
        Returns
        -------
        res : list
            A list containing all the odors
        """

        res = []
        for i in self.param['odors']:
            res.append(self._create_odor(odor_params[i], i))
        i = 0
        while len(res) < self.param['num_odors']:
            res.append(self._create_odor(odor_params['default'], f'default_{i}'))
            i += 1
        return res

    def _odor_binding_rate_permutation(self):
        """Permutes the binding rate of the odors, all according to the same random
        shuffle
        """
        not_default_shuffle = np.arange(self.param['num_glomeruli'])
        for i in self._odors:
            if i.name != 'default':
                i.shuffle_binding_rates(not_default_shuffle)
            else:
                i.shuffle_binding_rates()

    def _compute_hill_exponential(self, param):
        """Computes a hill exponential according to some parameters
        Parameters
        ----------
        param : dict
            A dictionary containing all the parameters necessary to compute the hill
            exponential
        Returns
        -------
        res : double
            The hill exponential
        """

        if isinstance(param, dict):
            res = np.random.uniform(param['min'], param['max'], self.param['num_glomeruli'])
        else:
            res = param
        return res

    @property
    def odors(self):
        """Getter for odors
        Returns
        -------
        res : list
            The list of odors
        """

        return self._odors

    @property
    def starting_concentration(self):
        """Getter for the starting concentration
        Returns
        -------
        res : double
            The starting concentration
        """

        return self.param['concentration']['start']

    @property
    def dilution_factor(self):
        """Getter for the dilution factor
        Returns
        -------
        res : double
            The dilution factor
        """

        return np.power(self.param['concentration']['dilution_factor']['base'], self.param['concentration']['dilution_factor']['exponent'])

    @property
    def event_duration(self):
        """Getter for the event duration in milliseconds
        Returns
        -------
        res : double
            The event duration
        """

        return self.param['event_duration']

    @property
    def resting_duration(self):
        """Getter for the resting duration in milliseconds
        Returns
        -------
        res : double
            The resting duration
        """

        return self.param['resting_duration']

    @property
    def hill_exponential(self):
        """Getter for the hill exponential
        Returns
        -------
        res : double
            The hill exponential
        """

        return self.param['hill_exponential']


    def assign_channel_to_events(self):
        """Assigns each event to a possible channel, assuring that at most one
        odor is presented to one channel during a time step
        """

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
        """Generates the connectivity matrix necessary for generating the inhibitory
        connections between some populations of neurons
        Parameters
        ----------
        connnectivity_type : str
            How to compute the connectivity_matrix
        self_inhibition : bool
            Whether a neuron can inhibit itself
        Returns
        -------
        connectivity_matrix : np.ndArray
            The connectivity matrix for neuron inhibitory connections
        """

        binding_rates_matrix = np.array([i.binding_rates for i in self._odors])
        if connectivity_type == 'correlation':
            connectivity_matrix = np.corrcoef(binding_rates_matrix,rowvar=False)
            connectivity_matrix = (connectivity_matrix + 1.0)/20.0
        elif connectivity_type == 'covariance':
            connectivity_matrix= np.cov(binding_rates_matrix,rowvar=False)
            connectivity_matrix= np.maximum(0.0, connectivity_matrix)
        elif connectivity_type == 'homogeneous':
            connectivity_matrix = np.ones((binding_rates_matrix.shape[1],
                                            binding_rates_matrix.shape[1]))
        else:
            connectivity_matrix = np.zeros((binding_rates_matrix.shape[1],
                                            binding_rates_matrix.shape[1]))

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

    @property
    def simulation_time(self):
        """Getter for the simulation time in milliseconds
        Returns
        -------
        res : double
            The total simulation time
        """

        max_time = max(self.events, key = lambda x : x['t_end'])['t_end']
        return max_time + self.param['resting_duration']


    @abstractmethod
    def _event_generation(self):
        """How an event is generated"""
        pass

    @abstractmethod
    def events_generation(self):
        """How all events are generated"""
        pass

    def get_events_for_channel(self):
        """Transforms all the events so to have a list of events for each
        possible channel
        Returns
        -------
        res : list
            A list where each element is the list of events happening on that channel
        """

        res = [[] for i in range(self.param['num_channels'])]

        for i in self.events:
            res[i['channel']].append(i)

        for i in res:
            i.sort(key = lambda x : x['t_start'])

        return res

    def __init__(self, param):
        """Generates the protocol class that contains all the events that happen
        during a simulation
        Parameters
        ----------
        param : dict
            A dictionary containing all the parameters necessary for building the
            protocol
        """

        self.param = param
        self._odors = self._create_odors(self.param['odors'])
        self._odor_binding_rate_permutation()
        self.param['hill_exponential'] = self._compute_hill_exponential(self.param['hill_exponential'])
        self.events = []
        self.connectivity_matrix = self._generate_inhibitory_connectivity(param['connectivity_type'], param['self_inhibition'])
